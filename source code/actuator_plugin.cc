/*
 * Actuator Plugin 
 *
 * - joint_param.csv is loaded ONCE in Load()
 */

#ifndef __GAZEBO_ACTUATOR_PLUGIN_STRICT_HH__
#define __GAZEBO_ACTUATOR_PLUGIN_STRICT_HH__

#include "gazebo/common/common.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/gazebo.hh"
#include <gazebo/physics/ode/ODEJoint.hh>
#include <ros/package.h>

#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace {
    // --------- small utilities ----------
    // Trim space in a string (both ends)

    constexpr double Pi = 3.1415926;
    constexpr double max_power = 100.0;  // in voltage units; adjust as needed
    constexpr double max_Theta = 0.79;
    constexpr double max_F     = 20.0;

    inline std::string Trim(std::string s) {
        auto is_ws = [](unsigned char ch){ return std::isspace(ch) != 0; };
        while (!s.empty() && is_ws(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
        while (!s.empty() && is_ws(static_cast<unsigned char>(s.back())))  s.pop_back();
        return s;
    }

    inline std::vector<std::string> Split_CSV(const std::string& line) {
        std::istringstream iss(line);
        std::vector<std::string> cols;
        std::string tok;
        while (std::getline(iss, tok, ',')) cols.push_back(Trim(tok));
        return cols;
    }

    inline double Sanitize(double v) {
        return std::isfinite(v) ? v : 0.0;
    }

    inline double Parse_Double(const std::string& s, int line_no, int col_no) {
        const std::string t = Trim(s);
        if (t.empty()) {
            throw std::runtime_error("[Actuator] CSV parse error: empty numeric field at line " +
                                    std::to_string(line_no) + ", col " + std::to_string(col_no));
        }
        size_t idx = 0;
        double v = 0.0;
        try {
            v = std::stod(t, &idx);
        } catch (const std::exception& e) {
            throw std::runtime_error("[Actuator] CSV parse error: invalid number '" + t +
                                    "' at line " + std::to_string(line_no) +
                                    ", col " + std::to_string(col_no) + " (" + e.what() + ")");
        }
        if (idx != t.size()) {
            throw std::runtime_error("[Actuator] CSV parse error: trailing chars in number '" + t +
                                    "' at line " + std::to_string(line_no) +
                                    ", col " + std::to_string(col_no));
        }
        if (!std::isfinite(v)) {
            throw std::runtime_error("[Actuator] CSV parse error: non-finite number '" + t +
                                    "' at line " + std::to_string(line_no) +
                                    ", col " + std::to_string(col_no));
        }
        return v;
    }

    inline int64_t SafeRoundStep(double t, double dt) {
        if (!(dt > 0.0) || !std::isfinite(dt)) return 0;
        return static_cast<int64_t>(std::llround(t / dt));
    }

    inline double Clamp(double v, double lim) {
        if (!std::isfinite(v)) return 0.0;
        if (v >  lim) return  lim;
        if (v < -lim) return -lim;
        return v;
    }

    constexpr double ALPHA_LIM = max_Theta * 4.0 * Pi * Pi * max_F * max_F;  // 12475.2
    inline void AccelClampTorque(const gazebo::physics::JointPtr& joint,
                                 double tz_raw,  double& tz_out, double alpha_lim) 
    {
        tz_raw = Sanitize(tz_raw);
        tz_out = 0.0;
        if (!joint) return;
        
        auto parent = joint->GetParent();        auto child = joint->GetChild();
        if (!parent || !child) {
            throw std::runtime_error("[Actuator] Joint '" + joint->GetName() + "' missing parent/child link.");
        }

        double izz_a = 0.0, izz_p = 0.0;
        if (auto inert = parent->GetInertial()) {
            izz_a = inert->IZZ();
        }
        if (auto inert = child->GetInertial()) {
            izz_p = inert->IZZ();
        }
        constexpr double IZZ_EPS = 1e-12;
        double izz_eff = std::min(izz_p, izz_a);
        if (std::isfinite(izz_eff)) {
            izz_eff = std::max(izz_eff, IZZ_EPS);
            tz_out = Clamp(Sanitize(tz_raw), alpha_lim * izz_eff);
        }
        else{
            throw std::runtime_error("[Actuator] Joint '" + joint->GetName() + "' has invalid moment of inertia for acceleration clamp.");
        }
        constexpr double T_HARD = 1e10;
        tz_out = Clamp(tz_out, T_HARD); // hard fuse
    }
}// anonymous namespace

namespace gazebo {

    class ActuatorPlugin : public ModelPlugin {
        public:
        ActuatorPlugin() = default;

        ~ActuatorPlugin() {
            if (act_out_.is_open())  act_out_.close();
            if (volt_out_.is_open()) volt_out_.close();
        }

        void Load(physics::ModelPtr _model, sdf::ElementPtr /*_sdf*/) override {
            model_ = _model;
            if (!model_) {
            throw std::runtime_error("[Actuator] Load(): model is null.");
            }
            ////////////////////////////////////////////////////////////////////////////
            // for (auto &j: model_->GetJoints()) {
            //     auto odej = boost::dynamic_pointer_cast<gazebo::physics::ODEJoint>(j);
            //     if (!odej){
            //         std::runtime_error("[Actuator] Joint '" + j->GetName() + "' is not an ODEJoint. All joints must be ODEJoint for this plugin.\n");
            //         continue;
            //     }
            //     std::cout<<" [Check] "<<j->GetName()<<" UsesImplicitSpringDamper= "<<odej->UsesImplicitSpringDamper()<<"\n";
            // }
            //////////////////////////////////////////////////////////////////////////

            // Step size from physics engine (fallback to your original 0.00025)
            step_size_ = 0.001;//0.00025;
            if (auto world = model_->GetWorld()) {
                if (auto phy = world->Physics()) {
                    const double dt = phy->GetMaxStepSize();
                    if (dt > 0.0 && std::isfinite(dt)) step_size_ = dt;
                }
            }

            // Derive model_id (guard substr)
            const std::string model_name = model_->GetName();
            const std::string model_id = (model_name.size() > 6) ? model_name.substr(6) : model_name;
            path_ = ros::package::getPath("des_mubot") + "/Database/" + model_id;

            // STRICT: load joint_param.csv and enforce 1-by-1 order/name match
            Load_Joint_Params(path_ + "/joint_param.csv");

            // STRICT: load control_signal.csv
            Load_Control_Signal(path_ + "/control_signal.csv");

            // Connect update
            updateConnection_ = event::Events::ConnectWorldUpdateBegin(
            boost::bind(&ActuatorPlugin::OnUpdate, this));

            std::cerr << "[Actuator] CSV and URDF matching\n";
        }

        private:
        
        private:
        // joint_param.csv row: name, ac_param, sf_param, stiff_ratio
        struct JointParam {
            double ac_param = 0.0;
            double sf_param = 0.0;
            double stiff_ratio = 0.0;
        };

        void Load_Joint_Params(const std::string& csv_path) {
            // 1) Get joint from Gazebo
            joints_ = model_->GetJoints();
            if (joints_.empty()) {
                throw std::runtime_error("[Actuator] Model has 0 joints.");
            }

            // Ensure model joint names are unique
            {
                std::unordered_set<std::string> seen;
                seen.reserve(joints_.size() * 2);
                for (const auto& j : joints_) {
                    if (!j) throw std::runtime_error("[Actuator] Null JointPtr in model->GetJoints().");
                    const std::string n = j->GetName();
                    if (n.empty()) throw std::runtime_error("[Actuator] Joint has empty name.");
                    if (!seen.insert(n).second) {
                        throw std::runtime_error("[Actuator] Duplicate joint name in model joints: '" + n + "'");
                    }
                }
            }

            // 2) Read CSV rows in order
            std::ifstream in(csv_path);
            if (!in.is_open()) {
                throw std::runtime_error("[Actuator] Cannot open CSV: " + csv_path);
            }

            std::vector<std::string> csv_names;
            std::vector<JointParam>  csv_params;
            csv_names.reserve(joints_.size());
            csv_params.reserve(joints_.size());

            std::unordered_set<std::string> csv_seen;
            csv_seen.reserve(joints_.size() * 2);

            std::string line;   int line_no = 0;
            while (std::getline(in, line)) {
                ++line_no;
                const std::string stripped = Trim(line);
                if (stripped.empty()) continue;
                const auto c = Split_CSV(stripped);
                // Pandas to_csv(header=False) with index gives 1 name + 3 values => 4 columns
                if (c.size() != 4) {
                    throw std::runtime_error("[Actuator] CSV format error: joint_param.csv expects exactly 4 columns "
                                            "(name, ac_param, sf_param, stiff_ratio). Got " +
                                            std::to_string(c.size()) + " at line " + std::to_string(line_no));
                }

                const std::string name = c[0];
                if (name.empty()) {
                    throw std::runtime_error("[Actuator] CSV format error: empty joint name at line " +
                                            std::to_string(line_no));
                }
                if (!csv_seen.insert(name).second) {
                    throw std::runtime_error("[Actuator] CSV format error: duplicate joint name '" + name +
                                            "' at line " + std::to_string(line_no));
                }

                JointParam p;
                p.sf_param     = Parse_Double(c[1], line_no, 1);
                p.ac_param     = Parse_Double(c[2], line_no, 2);
                p.stiff_ratio  = Parse_Double(c[3], line_no, 3);
                csv_names.push_back(name);      csv_params.push_back(p);
            }

            // STRICT: row count must match joint count exactly
            if (csv_names.size() != joints_.size()) {
            throw std::runtime_error("[Actuator] CSV/URDF mismatch: joint_param.csv rows = " +
                                    std::to_string(csv_names.size()) +
                                    ", model joints = " + std::to_string(joints_.size()) +
                                    ". They must match exactly.");
            }

            // STRICT: 1-by-1 order + exact name match
            for (size_t i = 0; i < joints_.size(); ++i) {
                const std::string jt_name = joints_[i]->GetName();
                const std::string csv_name = csv_names[i];
                if (jt_name != csv_name) {
                    throw std::runtime_error("[Actuator] CSV/URDF mismatch at index " + std::to_string(i) +
                                            ": model joint '" + jt_name + "' != csv '" + csv_name + "'");
                }
            }

            // Commit
            joint_names_ = std::move(csv_names);
            joint_params_ = std::move(csv_params);

            // Prepare arrays sized exactly to joint count
            const size_t N = joints_.size();
            amp_.assign(N, 0.0);            phase_.assign(N, 0.0);  voltage_.assign(N, 0.0);
            pos_.assign(N, 0.0);            vel_.assign(N, 0.0);
            torque_act_.assign(N, 0.0);     torque_spring_.assign(N, 0.0);     torque_rec_.assign(N, 0.0); // we keep it for logging; set to 0 (spring in URDF)
            
        }

        void Load_Control_Signal(const std::string& csv_path) {
            std::ifstream in(csv_path);
            if (!in.is_open()) {
                throw std::runtime_error("[Actuator] Cannot open CSV: " + csv_path);
            }

            // Read first non-empty line
            std::string line;
            while (std::getline(in, line)) {
                line = Trim(line);
                if (!line.empty()) break;
            }
            if (line.empty()) {
                throw std::runtime_error("[Actuator] control_signal.csv is empty: " + csv_path);
            }

            const auto c = Split_CSV(line);
            const size_t N = joints_.size();
            const size_t expected = 2 * N; // matches your current format
            if (c.size() != expected) {
            throw std::runtime_error("[Actuator] CSV format error: control_signal.csv expects exactly " +
                                    std::to_string(expected) + " columns, got " +
                                    std::to_string(c.size()) + ". File: " + csv_path);
            }

            std::vector<double> raw(expected, 0.0);
            for (size_t i = 0; i < expected; ++i) {
                raw[i] = Parse_Double(c[i], 1, static_cast<int>(i));
            }

            // Keep your original packing:
            // amp[0] = abs(raw[0]); phase[0] = 0;
            // for i>=1: amp[i] = abs(raw[2*i-1]); phase[i] = raw[2*i];
            // frequency = raw[2*N - 1]
            amp_[0] = std::fabs(raw[0]);
            phase_[0] = 0.0;

            double amp_sum = amp_[0];
            for (size_t i = 1; i < N; ++i) {
                amp_[i] = std::fabs(raw[2 * i - 1]);
                phase_[i] = raw[2 * i];
                amp_sum += amp_[i];
            }
            frequency_ = std::fabs(raw[2 * N - 1]);

            // Power limiting
            if (amp_sum > max_power) {
                const double ratio = max_power / amp_sum;
                for (size_t i = 0; i < N; ++i) amp_[i] *= ratio;
            }

            if (!std::isfinite(frequency_) || frequency_ <= 0.0) {
                throw std::runtime_error("[Actuator] Invalid frequency in control_signal.csv: " +
                                        std::to_string(frequency_));
            }
        }

        // void Files_Exists() {
        //     if (output_files_initialized_) return;

        //     // Match your original behavior: create/overwrite at the first record step
        //     {
        //         std::ofstream out(path_ + "/actuator_output.csv", std::ios::out);
        //         if (!out.is_open()) throw std::runtime_error("[Actuator] Cannot create actuator_output.csv");
        //     }
        //     {
        //         std::ofstream out(path_ + "/voltage_output.csv", std::ios::out);
        //         if (!out.is_open()) throw std::runtime_error("[Actuator] Cannot create voltage_output.csv");
        //     }

        //     output_files_initialized_ = true;
        // }

        void Files_Exists() {          // ← member function, inside class
            if (output_files_initialized_) return;
            act_out_.open(path_ + "/actuator_output.csv", std::ios::out);
            act_out_ << std::scientific << std::setprecision(6);
            if (!act_out_.is_open())
                throw std::runtime_error("[Actuator] Cannot create actuator_output.csv");
            volt_out_.open(path_ + "/voltage_output.csv", std::ios::out);
            volt_out_ << std::scientific << std::setprecision(6);
            if (!volt_out_.is_open())
                throw std::runtime_error("[Actuator] Cannot create voltage_output.csv");
            output_files_initialized_ = true;
        }

        void OnUpdate() {
            if (!model_) return;

            const double simu_time = model_->GetWorld()->SimTime().Double();
            const int64_t simu_step = SafeRoundStep(simu_time, step_size_);
            const int64_t record_step = SafeRoundStep(record_time_, step_size_);

            const size_t N = joints_.size();
            if (N == 0) return;
            std::fill(pos_.begin(), pos_.end(), 0.0);                   std::fill(vel_.begin(), vel_.end(), 0.0);                        std::fill(voltage_.begin(), voltage_.end(), 0.0);   
            std::fill(torque_rec_.begin(), torque_rec_.end(), 0.0);     std::fill(torque_act_.begin(), torque_act_.end(), 0.0);

            // compute and apply actuator torques
            for (size_t i = 0; i < N; ++i) {
                auto j = joints_[i];
                if (!j) {
                    throw std::runtime_error("[Actuator] Null joint during OnUpdate at index " + std::to_string(i));
                }

                pos_[i] = j->Position(0);
                vel_[i] = j->GetVelocity(0);

                const double v = amp_[i] * std::sin(2.0 * Pi * (frequency_ * simu_time + phase_[i]));
                // clamp voltage
                voltage_[i] = Clamp(v, 15.0);

                const double ac = joint_params_[i].ac_param;

                // torque_act = (voltage - ac*vel)/90 * ac * tor_ratio
                torque_act_[i] = ((voltage_[i] - ac * vel_[i]) / 90.0) * ac * scaling;
                // torque_spring_[i] = - joint_params_[i].sf_param * pos_[i]; 
                torque_spring_[i] = 0.0; // we set it to 0 since spring is in URDF

                const double torque_raw = torque_act_[i] + torque_spring_[i];
                double torque_clamped = 0.0;
                AccelClampTorque(j, torque_raw, torque_clamped, ALPHA_LIM);
                torque_rec_[i] = torque_clamped; // for logging
                j->SetForce(0, torque_clamped); // actuator + spring
            }

            //////////////////////////////// recording after record_time_
            // if (simu_step >= record_step) {
            //     Files_Exists();

            //     // Keep original: do not write at the exact record_step (optional)
            //     if (simu_step > record_step) {
            //         // actuator_output.csv: time, (act, spring)*N
            //         {
            //         // std::ofstream out(path_ + "/actuator_output.csv", std::ios::app);
            //         if (!act_out_.is_open()) throw std::runtime_error("[Actuator] Cannot append actuator_output.csv");

            //         act_out_ << simu_time;
            //         for (size_t i = 0; i < N; ++i) {
            //             act_out_ << "," << torque_act_[i] << "," << torque_spring_[i] << "," << torque_rec_[i];
            //         }
            //         act_out_ << "\n";
            //         }

            //         // voltage_output.csv: time, voltage*N
            //         {
            //         // std::ofstream out(path_ + "/voltage_output.csv", std::ios::app);
            //         if (!volt_out_.is_open()) throw std::runtime_error("[Actuator] Cannot append voltage_output.csv");

            //         volt_out_ << simu_time;
            //         for (size_t i = 0; i < N; ++i) {
            //             volt_out_ << "," << voltage_[i];
            //         }
            //         volt_out_ << "\n";
            //         }
            //     }
            // }
        }

        private:
        // Gazebo joint and link pointers (cached for efficiency; assumes model structure does not change at runtime)
        std::vector<physics::JointPtr> joints_;
        std::vector<physics::LinkPtr> links_;
        physics::ModelPtr model_;
        event::ConnectionPtr updateConnection_;

        std::string path_;

        double step_size_ = 0.001;//0.00025;
        const double record_time_ = 4.0;
        const double stop_time_   = 6.1;
        const double scaling      = 1000.0;

        bool output_files_initialized_ = false;


        // joint names and params from CSV, in the same order as Gazebo joints
        std::vector<std::string> joint_names_;
        std::vector<JointParam> joint_params_;
        // Output files
        std::ofstream act_out_;
        std::ofstream volt_out_;
        // Control
        double frequency_ = 0.0;
        std::vector<double> amp_;
        std::vector<double> phase_;

        // Runtime / logging
        std::vector<double> voltage_;
        std::vector<double> pos_;
        std::vector<double> vel_;
        std::vector<double> torque_act_;
        std::vector<double> torque_rec_;
        std::vector<double> torque_spring_;
    };

    GZ_REGISTER_MODEL_PLUGIN(ActuatorPlugin)

} // namespace gazebo

#endif
