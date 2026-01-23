/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * @file export_simulation.cpp
 * @brief Exports OpenVINS simulation data to tiny-vio replay CSV format.
 *
 * This executable runs the OpenVINS VIO pipeline on simulated data (like run_simulation.cpp)
 * while also exporting the raw sensor data in a format compatible with tiny-vio's replay harness.
 *
 * The output includes:
 *   - replay.csv: IMU measurements and camera feature detections
 *   - groundtruth.csv: Ground truth poses at VIO-processed timestamps
 *   - features_3d.csv: 3D positions of all features in the map
 *   - calibration.yaml: Camera and IMU calibration parameters
 *
 * By running the actual VIO pipeline, we ensure ground truth is queried at timestamps that
 * are guaranteed to be within the simulator's bias history (VIO naturally lags behind).
 */

#include <csignal>
#include <fstream>
#include <iomanip>
#include <memory>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "sim/Simulator.h"
#include "state/State.h"
#include "utils/colors.h"
#include "utils/dataset_reader.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#if ROS_AVAILABLE == 1
#include <ros/ros.h>
#elif ROS_AVAILABLE == 2
#include <rclcpp/rclcpp.hpp>
#endif

using namespace ov_msckf;

// Global pointers for signal handler
std::shared_ptr<Simulator> sim;
std::shared_ptr<VioManager> sys;

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) { std::exit(signum); }

/**
 * @brief Convert seconds (double) to microseconds (uint64_t)
 */
inline uint64_t toMicroseconds(double seconds) {
  return static_cast<uint64_t>(seconds * 1e6);
}

/**
 * @brief Write IMU measurement line in tiny-vio format
 * Format: IMU,timestamp_us,wx,wy,wz,ax,ay,az
 */
void writeImuLine(std::ofstream &out, double ts, const Eigen::Vector3d &wm, const Eigen::Vector3d &am) {
  out << "IMU," << toMicroseconds(ts) << "," << std::setprecision(9) << wm(0) << "," << wm(1) << "," << wm(2) << "," << am(0) << ","
      << am(1) << "," << am(2) << "\n";
}

/**
 * @brief Write FRAME line in tiny-vio format
 * Format: FRAME,timestamp_us,frame_id
 */
void writeFrameLine(std::ofstream &out, double ts, uint32_t frame_id) {
  out << "FRAME," << toMicroseconds(ts) << "," << frame_id << "\n";
}

/**
 * @brief Write DET (detection) line in tiny-vio format
 * Format: DET,frame_id,u,v,track_id,led_id
 * Note: led_id=255 indicates blob-only mode (no LED constellation)
 */
void writeDetLine(std::ofstream &out, uint32_t frame_id, float u, float v, uint16_t track_id) {
  out << "DET," << frame_id << "," << std::setprecision(6) << u << "," << v << "," << track_id << ",255\n";
}

/**
 * @brief Write ground truth state line
 * Format: timestamp_us,qx,qy,qz,qw,px,py,pz,vx,vy,vz,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z
 */
void writeGroundTruthLine(std::ofstream &out, const Eigen::Matrix<double, 17, 1> &imustate) {
  uint64_t ts_us = toMicroseconds(imustate(0));
  // imustate format: [time, q_GtoI (x,y,z,w), p_IinG, v_IinG, b_gyro, b_accel]
  out << ts_us << "," << std::setprecision(9) << imustate(1) << "," // qx
      << imustate(2) << ","                                         // qy
      << imustate(3) << ","                                         // qz
      << imustate(4) << ","                                         // qw
      << imustate(5) << ","                                         // px
      << imustate(6) << ","                                         // py
      << imustate(7) << ","                                         // pz
      << imustate(8) << ","                                         // vx
      << imustate(9) << ","                                         // vy
      << imustate(10) << ","                                        // vz
      << imustate(11) << ","                                        // bg_x
      << imustate(12) << ","                                        // bg_y
      << imustate(13) << ","                                        // bg_z
      << imustate(14) << ","                                        // ba_x
      << imustate(15) << ","                                        // ba_y
      << imustate(16) << "\n";                                      // ba_z
}

/**
 * @brief Write feature map to CSV
 * Format: feature_id,x,y,z
 */
void writeFeatureMap(const std::string &path, const std::unordered_map<size_t, Eigen::Vector3d> &featmap) {
  std::ofstream out(path);
  if (!out.is_open()) {
    PRINT_ERROR(RED "Failed to open feature map file: %s\n" RESET, path.c_str());
    return;
  }

  out << "feature_id,x,y,z\n";
  for (const auto &feat : featmap) {
    out << feat.first << "," << std::setprecision(9) << feat.second(0) << "," << feat.second(1) << "," << feat.second(2) << "\n";
  }
  out.close();
  PRINT_INFO("Wrote %zu features to %s\n", featmap.size(), path.c_str());
}

/**
 * @brief Write calibration parameters to YAML
 */
void writeCalibration(const std::string &path, const VioManagerOptions &params) {
  std::ofstream out(path);
  if (!out.is_open()) {
    PRINT_ERROR(RED "Failed to open calibration file: %s\n" RESET, path.c_str());
    return;
  }

  out << std::setprecision(12);
  out << "# Calibration parameters exported from OpenVINS simulation\n";
  out << "# Generated by export_simulation\n\n";

  // Camera parameters
  for (int i = 0; i < params.state_options.num_cameras; i++) {
    auto cam = params.camera_intrinsics.at(i);
    Eigen::VectorXd intrinsics = cam->get_value();
    Eigen::VectorXd extrinsics = params.camera_extrinsics.at(i);

    out << "camera" << i << ":\n";
    out << "  resolution: [" << cam->w() << ", " << cam->h() << "]\n";
    out << "  intrinsics: [" << intrinsics(0) << ", " << intrinsics(1) << ", " << intrinsics(2) << ", " << intrinsics(3) << "]\n";
    out << "  distortion: [" << intrinsics(4) << ", " << intrinsics(5) << ", " << intrinsics(6) << ", " << intrinsics(7) << "]\n";

    // Extrinsics: q_ItoC, p_IinC
    out << "  extrinsics:\n";
    out << "    q_ItoC: [" << extrinsics(0) << ", " << extrinsics(1) << ", " << extrinsics(2) << ", " << extrinsics(3) << "]\n";
    out << "    p_IinC: [" << extrinsics(4) << ", " << extrinsics(5) << ", " << extrinsics(6) << "]\n";
    out << "\n";
  }

  // IMU parameters
  out << "imu:\n";
  out << "  gyro_noise_density: " << params.imu_noises.sigma_w << "\n";
  out << "  gyro_random_walk: " << params.imu_noises.sigma_wb << "\n";
  out << "  accel_noise_density: " << params.imu_noises.sigma_a << "\n";
  out << "  accel_random_walk: " << params.imu_noises.sigma_ab << "\n";
  out << "  gravity_magnitude: " << params.gravity_mag << "\n";
  out << "\n";

  // Timing
  out << "timing:\n";
  out << "  cam_imu_dt: " << params.calib_camimu_dt << "\n";
  out << "  imu_freq_hz: " << params.sim_freq_imu << "\n";
  out << "  cam_freq_hz: " << params.sim_freq_cam << "\n";

  out.close();
  PRINT_INFO("Wrote calibration to %s\n", path.c_str());
}

int main(int argc, char **argv) {

  // Ensure we have a path, if the user passes it then we should use it
  std::string config_path = "unset_path_to_config.yaml";
  std::string output_dir = "./sim_export";
  if (argc > 1) {
    config_path = argv[1];
  }
  if (argc > 2) {
    output_dir = argv[2];
  }

#if ROS_AVAILABLE == 1
  // Launch our ros node
  ros::init(argc, argv, "export_simulation");
  auto nh = std::make_shared<ros::NodeHandle>("~");
  nh->param<std::string>("config_path", config_path, config_path);
  nh->param<std::string>("output_dir", output_dir, output_dir);
#elif ROS_AVAILABLE == 2
  // Launch our ros node
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<rclcpp::Node>("export_simulation", options);
  node->get_parameter<std::string>("config_path", config_path);
  node->get_parameter<std::string>("output_dir", output_dir);
#endif

  // Load the config
  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
#if ROS_AVAILABLE == 1
  parser->set_node_handler(nh);
#elif ROS_AVAILABLE == 2
  parser->set_node(node);
#endif

  // Verbosity
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  // Create our VIO system options
  VioManagerOptions params;
  params.print_and_load(parser);
  params.print_and_load_simulation(parser);
  params.num_opencv_threads = 0; // for repeatability
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;

  // Ensure we read in all parameters required
  if (!parser->successful()) {
    PRINT_ERROR(RED "Unable to parse all parameters, please fix\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Create simulator and VIO system
  sim = std::make_shared<Simulator>(params);
  sys = std::make_shared<VioManager>(params);

  //===================================================================================
  // Initialize VIO with ground truth (same as run_simulation.cpp)
  //===================================================================================

  // Get initial state at next IMU timestep (ensures bias history covers it)
  double next_imu_time = sim->current_timestamp() + 1.0 / params.sim_freq_imu;
  Eigen::Matrix<double, 17, 1> imustate_init;
  bool success = sim->get_state(next_imu_time, imustate_init);
  if (!success) {
    PRINT_ERROR(RED "[SIM]: Could not initialize the filter to the first state\n" RESET);
    PRINT_ERROR(RED "[SIM]: Did the simulator load properly???\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Adjust for camera-IMU time offset
  imustate_init(0, 0) -= sim->get_true_parameters().calib_camimu_dt;

  // Initialize VIO with ground truth
  sys->initialize_with_gt(imustate_init);

  //===================================================================================
  // Open output files
  //===================================================================================

  // Create output directory
  std::string mkdir_cmd = "mkdir -p " + output_dir;
  int ret = system(mkdir_cmd.c_str());
  if (ret != 0) {
    PRINT_WARNING(YELLOW "mkdir command returned non-zero: %d\n" RESET, ret);
  }

  // Open replay CSV
  std::string replay_path = output_dir + "/replay.csv";
  std::ofstream replay_out(replay_path);
  if (!replay_out.is_open()) {
    PRINT_ERROR(RED "Failed to open replay file: %s\n" RESET, replay_path.c_str());
    std::exit(EXIT_FAILURE);
  }

  // Open ground truth CSV
  std::string gt_path = output_dir + "/groundtruth.csv";
  std::ofstream gt_out(gt_path);
  if (!gt_out.is_open()) {
    PRINT_ERROR(RED "Failed to open ground truth file: %s\n" RESET, gt_path.c_str());
    std::exit(EXIT_FAILURE);
  }
  gt_out << "timestamp_us,qx,qy,qz,qw,px,py,pz,vx,vy,vz,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n";

  PRINT_INFO("===========================================\n");
  PRINT_INFO("EXPORTING SIMULATION TO TINY-VIO FORMAT\n");
  PRINT_INFO("===========================================\n");
  PRINT_INFO("Output directory: %s\n", output_dir.c_str());
  PRINT_INFO("Number of cameras: %d (exporting camera 0 only for monocular)\n", params.state_options.num_cameras);
  PRINT_INFO("IMU frequency: %.1f Hz\n", params.sim_freq_imu);
  PRINT_INFO("Camera frequency: %.1f Hz\n", params.sim_freq_cam);

  //===================================================================================
  // Main simulation loop (mirrors run_simulation.cpp but also exports data)
  //===================================================================================

  uint32_t frame_id = 0;
  uint64_t imu_count = 0;
  uint64_t det_count = 0;
  uint64_t gt_count = 0;

  // Buffer for camera data (VIO processes previous frame, like run_simulation.cpp)
  double buffer_timecam = -1;
  std::vector<int> buffer_camids;
  std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> buffer_feats;

  signal(SIGINT, signal_callback_handler);

#if ROS_AVAILABLE == 1
  while (sim->ok() && ros::ok()) {
#elif ROS_AVAILABLE == 2
  while (sim->ok() && rclcpp::ok()) {
#else
  while (sim->ok()) {
#endif

    // IMU: get the next simulated IMU measurement if we have it
    ov_core::ImuData message_imu;
    bool hasimu = sim->get_next_imu(message_imu.timestamp, message_imu.wm, message_imu.am);
    if (hasimu) {
      // Export IMU data for TinyVIO
      writeImuLine(replay_out, message_imu.timestamp, message_imu.wm, message_imu.am);
      imu_count++;

      // Feed to VIO system
      sys->feed_measurement_imu(message_imu);
    }

    // CAM: get the next simulated camera uv measurements if we have them
    double time_cam;
    std::vector<int> camids;
    std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
    bool hascam = sim->get_next_cam(time_cam, camids, feats);

    if (hascam) {
      // Export current camera frame for TinyVIO
      writeFrameLine(replay_out, time_cam, frame_id);

      // Write detections for camera 0 only (monocular mode)
      if (!feats.empty() && !feats[0].empty()) {
        for (const auto &feat : feats[0]) {
          uint16_t track_id = static_cast<uint16_t>(feat.first % 65536);
          writeDetLine(replay_out, frame_id, feat.second(0), feat.second(1), track_id);
          det_count++;
        }
      }

      // Feed PREVIOUS frame to VIO (1-frame buffer, same as run_simulation.cpp)
      if (buffer_timecam != -1) {
        sys->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);

        // Query ground truth at VIO's processed timestamp (guaranteed to be within bias history)
        if (sys->initialized()) {
          double gt_time = sys->get_state()->_timestamp + sim->get_true_parameters().calib_camimu_dt;
          Eigen::Matrix<double, 17, 1> imustate;
          if (sim->get_state(gt_time, imustate)) {
            writeGroundTruthLine(gt_out, imustate);
            gt_count++;
          }
        }
      }

      // Buffer current frame for next iteration
      buffer_timecam = time_cam;
      buffer_camids = camids;
      buffer_feats = feats;
      frame_id++;
    }
  }

  // Process the last buffered frame
  if (buffer_timecam != -1) {
    sys->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);
    if (sys->initialized()) {
      double gt_time = sys->get_state()->_timestamp + sim->get_true_parameters().calib_camimu_dt;
      Eigen::Matrix<double, 17, 1> imustate;
      if (sim->get_state(gt_time, imustate)) {
        writeGroundTruthLine(gt_out, imustate);
        gt_count++;
      }
    }
  }

  //===================================================================================
  // Finalize and write additional files
  //===================================================================================

  replay_out.close();
  gt_out.close();

  // Write feature map
  std::string featmap_path = output_dir + "/features_3d.csv";
  writeFeatureMap(featmap_path, sim->get_map());

  // Write calibration
  std::string calib_path = output_dir + "/calibration.yaml";
  writeCalibration(calib_path, sim->get_true_parameters());

  PRINT_INFO("===========================================\n");
  PRINT_INFO("EXPORT COMPLETE\n");
  PRINT_INFO("===========================================\n");
  PRINT_INFO("Frames exported: %u\n", frame_id);
  PRINT_INFO("IMU samples: %lu\n", imu_count);
  PRINT_INFO("Ground truth states: %lu\n", gt_count);
  PRINT_INFO("Detections: %lu\n", det_count);
  PRINT_INFO("Files written:\n");
  PRINT_INFO("  - %s\n", replay_path.c_str());
  PRINT_INFO("  - %s\n", gt_path.c_str());
  PRINT_INFO("  - %s\n", featmap_path.c_str());
  PRINT_INFO("  - %s\n", calib_path.c_str());

#if ROS_AVAILABLE == 1
  ros::shutdown();
#elif ROS_AVAILABLE == 2
  rclcpp::shutdown();
#endif

  return EXIT_SUCCESS;
}
