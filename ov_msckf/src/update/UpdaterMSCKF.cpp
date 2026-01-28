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

#include "UpdaterMSCKF.h"

#include "UpdaterHelper.h"

#include "feat/Feature.h"
#include "feat/FeatureInitializer.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/LandmarkRepresentation.h"
#include "utils/colors.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/distributions/chi_squared.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterMSCKF::UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options) : _options(options) {

  // Save our raw pixel noise squared
  _options.sigma_pix_sq = std::pow(_options.sigma_pix, 2);

  // Save our feature initializer
  initializer_feat = std::shared_ptr<ov_core::FeatureInitializer>(new ov_core::FeatureInitializer(feat_init_options));

  // Initialize the chi squared test table with confidence level 0.95
  // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
  for (int i = 1; i < 500; i++) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_table[i] = boost::math::quantile(chi_squared_dist, 0.95);
  }
}

void UpdaterMSCKF::update(std::shared_ptr<State> state, std::vector<std::shared_ptr<Feature>> &feature_vec) {

  // Return if no features
  if (feature_vec.empty())
    return;

  // Start timing
  boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5;
  rT0 = boost::posix_time::microsec_clock::local_time();

  // 0. Get all timestamps our clones are at (and thus valid measurement times)
  std::vector<double> clonetimes;
  for (const auto &clone_imu : state->_clones_IMU) {
    clonetimes.emplace_back(clone_imu.first);
  }

  // Track selection logging for TinyVIO comparison
  PRINT_DEBUG("[TRACK_SELECT] n_input=%zu n_clones=%zu\n", feature_vec.size(), clonetimes.size());

  // 1. Clean all feature measurements and make sure they all have valid clone times
  auto it0 = feature_vec.begin();
  while (it0 != feature_vec.end()) {

    // Clean the feature
    (*it0)->clean_old_measurements(clonetimes);

    // Count how many measurements
    int ct_meas = 0;
    for (const auto &pair : (*it0)->timestamps) {
      ct_meas += (*it0)->timestamps[pair.first].size();
    }

    // Log track details for comparison
    PRINT_DEBUG("[FEATDB_FILTER] track=%zu num_obs=%d distinct_clones=%d\n",
                (*it0)->featid, ct_meas, ct_meas);

    // Per-track detailed logging
    PRINT_DEBUG("[TRACK_SELECT]   feat_id=%zu n_obs=%d timestamps=[", (*it0)->featid, ct_meas);
    for (const auto &pair : (*it0)->timestamps) {
      for (size_t ti = 0; ti < pair.second.size(); ++ti) {
        PRINT_DEBUG("%.6f%s", pair.second[ti], (ti < pair.second.size()-1) ? "," : "");
      }
    }
    PRINT_DEBUG("]\n");

    // Remove if we don't have enough
    if (ct_meas < 2) {
      PRINT_DEBUG("[FEATDB_FILTER] track=%zu REJECT reason=insufficient_clones (have=%d need=2)\n",
                  (*it0)->featid, ct_meas);
      (*it0)->to_delete = true;
      it0 = feature_vec.erase(it0);
    } else {
      it0++;
    }
  }
  rT1 = boost::posix_time::microsec_clock::local_time();

  // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  for (const auto &clone_calib : state->_calib_IMUtoCAM) {

    // For this camera, create the vector of camera poses
    std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    for (const auto &clone_imu : state->_clones_IMU) {

      // Get current camera pose
      Eigen::Matrix<double, 3, 3> R_GtoCi = clone_calib.second->Rot() * clone_imu.second->Rot();
      Eigen::Matrix<double, 3, 1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose() * clone_calib.second->pos();

      // Append to our map
      clones_cami.insert({clone_imu.first, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
    }

    // Append to our map
    clones_cam.insert({clone_calib.first, clones_cami});
  }

  // Log clone poses for TinyVIO comparison (only first camera)
  if (!clones_cam.empty()) {
    auto& clones_cam0 = clones_cam.begin()->second;
    int clone_idx = 0;
    for (auto& clone_pair : clones_cam0) {
      Eigen::Vector3d p = clone_pair.second.pos();
      PRINT_DEBUG("[MSCKF_CLONES] clone[%d] ts=%.6f p=[%.6f,%.6f,%.6f]\n",
                  clone_idx++, clone_pair.first, p(0), p(1), p(2));
    }
  }

  // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
  auto it1 = feature_vec.begin();
  while (it1 != feature_vec.end()) {

    // Triangulate the feature and remove if it fails
    bool success_tri = true;
    if (initializer_feat->config().triangulate_1d) {
      success_tri = initializer_feat->single_triangulation_1d(*it1, clones_cam);
    } else {
      success_tri = initializer_feat->single_triangulation(*it1, clones_cam);
    }

    // Gauss-newton refine the feature
    bool success_refine = true;
    if (initializer_feat->config().refine_features) {
      success_refine = initializer_feat->single_gaussnewton(*it1, clones_cam);
    }

    // Remove the feature if not a success
    if (!success_tri || !success_refine) {
      PRINT_DEBUG("[FEATDB_FILTER] track=%zu REJECT reason=triangulation_failed\n", (*it1)->featid);
      (*it1)->to_delete = true;
      it1 = feature_vec.erase(it1);
      continue;
    }
    // Comparison logging for TinyVIO parity verification
    // Get first and last UV for matching tracks between systems
    size_t cam_id = (*it1)->uvs.begin()->first;
    const auto& uvs_vec = (*it1)->uvs.at(cam_id);
    int num_obs = uvs_vec.size();
    Eigen::VectorXf first_uv = uvs_vec.front();
    Eigen::VectorXf last_uv = uvs_vec.back();
    PRINT_DEBUG("[MSCKF_TRI] track=%zu pos=[%.6f,%.6f,%.6f] num_obs=%d first_uv=[%.2f,%.2f] last_uv=[%.2f,%.2f]\n",
                (*it1)->featid, (*it1)->p_FinG(0), (*it1)->p_FinG(1), (*it1)->p_FinG(2),
                num_obs, first_uv(0), first_uv(1), last_uv(0), last_uv(1));

    // MSCKF_FEAT: Feature selection and triangulation result (for parity comparison)
    PRINT_DEBUG("[MSCKF_FEAT] feat_id=%zu num_obs=%d p_FinG=[%e,%e,%e]\n",
                (*it1)->featid, num_obs, (*it1)->p_FinG(0), (*it1)->p_FinG(1), (*it1)->p_FinG(2));
#ifdef OPENVINS_PARITY_DEBUG
    // High-precision parity debug output for triangulation
    PRINT_DEBUG("[PARITY_TRI] feat_id=%zu\n", (*it1)->featid);
    PRINT_DEBUG("[PARITY_TRI] num_obs=%d\n", num_obs);
    PRINT_DEBUG("[PARITY_TRI] p_FinG=%.17g,%.17g,%.17g\n",
        (*it1)->p_FinG(0), (*it1)->p_FinG(1), (*it1)->p_FinG(2));
    PRINT_DEBUG("[PARITY_TRI] uv_first=%.17g,%.17g\n", (double)first_uv(0), (double)first_uv(1));
    PRINT_DEBUG("[PARITY_TRI] uv_last=%.17g,%.17g\n", (double)last_uv(0), (double)last_uv(1));
    // Clone timestamps used for this feature
    {
      std::string ts_str = "[PARITY_TRI] clone_ts=";
      const auto& ts_vec = (*it1)->timestamps.at(cam_id);
      for (size_t ti = 0; ti < ts_vec.size(); ti++) {
        char buf[48];
        snprintf(buf, sizeof(buf), "%.17g%s", ts_vec[ti], ti < ts_vec.size()-1 ? "," : "\n");
        ts_str += buf;
      }
      PRINT_DEBUG("%s", ts_str.c_str());
    }
#endif
    it1++;
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Calculate the max possible measurement size
  size_t max_meas_size = 0;
  for (size_t i = 0; i < feature_vec.size(); i++) {
    for (const auto &pair : feature_vec.at(i)->timestamps) {
      max_meas_size += 2 * feature_vec.at(i)->timestamps[pair.first].size();
    }
  }

  // Calculate max possible state size (i.e. the size of our covariance)
  // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
  size_t max_hx_size = state->max_covariance_size();
  for (auto &landmark : state->_features_SLAM) {
    max_hx_size -= landmark.second->size();
  }

  // Large Jacobian and residual of *all* features for this update
  Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
  Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
  std::unordered_map<std::shared_ptr<Type>, size_t> Hx_mapping;
  std::vector<std::shared_ptr<Type>> Hx_order_big;
  size_t ct_jacob = 0;
  size_t ct_meas = 0;

  // DEBUG: Log allocated matrix sizes
  PRINT_DEBUG("[MATRIX_ALLOC] n_features=%zu max_meas_size=%zu max_hx_size=%zu\n",
              feature_vec.size(), max_meas_size, max_hx_size);
  PRINT_DEBUG("[MATRIX_ALLOC] Hx_big allocated: %zu rows x %zu cols (%.2f MB)\n",
              max_meas_size, max_hx_size, (max_meas_size * max_hx_size * 8.0) / (1024.0 * 1024.0));

  // 4. Compute linear system for each feature, nullspace project, and reject
  auto it2 = feature_vec.begin();
  while (it2 != feature_vec.end()) {

    // Convert our feature into our current format
    UpdaterHelper::UpdaterHelperFeature feat;
    feat.featid = (*it2)->featid;
    feat.uvs = (*it2)->uvs;
    feat.uvs_norm = (*it2)->uvs_norm;
    feat.timestamps = (*it2)->timestamps;

    // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
    feat.feat_representation = state->_options.feat_rep_msckf;
    if (state->_options.feat_rep_msckf == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
    }

    // Save the position and its fej value
    if (LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
      feat.anchor_cam_id = (*it2)->anchor_cam_id;
      feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
      feat.p_FinA = (*it2)->p_FinA;
      feat.p_FinA_fej = (*it2)->p_FinA;
    } else {
      feat.p_FinG = (*it2)->p_FinG;
      feat.p_FinG_fej = (*it2)->p_FinG;
    }

    // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd res;
    std::vector<std::shared_ptr<Type>> Hx_order;

    // Get the Jacobian for this feature
    UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

    // Log per-feature clone observations (for TinyVIO parity comparison)
    PRINT_DEBUG("[FEAT_CLONES] feat=%zu n_vars=%zu\n", (*it2)->featid, Hx_order.size());

    // Nullspace project
    UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

    /// Chi2 distance check
    Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd S = H_x * P_marg * H_x.transpose();
    S.diagonal() += _options.sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
    double chi2 = res.dot(S.llt().solve(res));

    // Get our threshold (we precompute up to 500 but handle the case that it is more)
    double chi2_check;
    if (res.rows() < 500) {
      chi2_check = chi_squared_table[res.rows()];
    } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
      PRINT_WARNING(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
    }

    // Chi-squared gating logging for TinyVIO comparison
    PRINT_DEBUG("[CHI2_GATE] feat_id=%zu dof=%d chi2=%.6f thresh=%.6f %s\n",
                (*it2)->featid, (int)res.rows(), chi2, _options.chi2_multipler * chi2_check,
                (chi2 > _options.chi2_multipler * chi2_check) ? "FAIL" : "PASS");
    PRINT_DEBUG("[CHI2_GATE]   res_norm=%.6f res_first=[%.9f, %.9f]\n",
                res.norm(), res(0), (res.rows() > 1) ? res(1) : 0.0);

    // MSCKF_CHI2: Chi-squared gating result (for parity comparison)
    PRINT_DEBUG("[MSCKF_CHI2] chi2=%e thresh=%e pass=%d\n",
                chi2, _options.chi2_multipler * chi2_check,
                (chi2 > _options.chi2_multipler * chi2_check) ? 0 : 1);
#ifdef OPENVINS_PARITY_DEBUG
    // High-precision parity debug output for chi-squared gating
    PRINT_DEBUG("[PARITY_CHI2] feat_id=%zu\n", (*it2)->featid);
    PRINT_DEBUG("[PARITY_CHI2] dof=%d\n", (int)res.rows());
    PRINT_DEBUG("[PARITY_CHI2] chi2_stat=%.17g\n", chi2);
    PRINT_DEBUG("[PARITY_CHI2] chi2_thresh=%.17g\n", _options.chi2_multipler * chi2_check);
    PRINT_DEBUG("[PARITY_CHI2] pass=%d\n", (chi2 > _options.chi2_multipler * chi2_check) ? 0 : 1);
#endif

    // Check if we should delete or not
    if (chi2 > _options.chi2_multipler * chi2_check) {
      PRINT_DEBUG("[FEATDB_FILTER] track=%zu REJECT reason=chi2_gate_failed\n", (*it2)->featid);
      (*it2)->to_delete = true;
      it2 = feature_vec.erase(it2);
      // PRINT_DEBUG("featid = %d\n", feat.featid);
      // PRINT_DEBUG("chi2 = %f > %f\n", chi2, _options.chi2_multipler*chi2_check);
      // std::stringstream ss;
      // ss << "res = " << std::endl << res.transpose() << std::endl;
      // PRINT_DEBUG(ss.str().c_str());
      continue;
    }

    // Log acceptance
    PRINT_DEBUG("[FEATDB_FILTER] track=%zu ACCEPT triangulated_pos=[%.6f,%.6f,%.6f]\n",
                (*it2)->featid, feat.p_FinG(0), feat.p_FinG(1), feat.p_FinG(2));

    // We are good!!! Append to our large H vector
    size_t ct_hx = 0;
    for (const auto &var : Hx_order) {

      // Ensure that this variable is in our Jacobian
      if (Hx_mapping.find(var) == Hx_mapping.end()) {
        Hx_mapping.insert({var, ct_jacob});
        Hx_order_big.push_back(var);
        ct_jacob += var->size();
      }

      // Append to our large Jacobian
      Hx_big.block(ct_meas, Hx_mapping[var], H_x.rows(), var->size()) = H_x.block(0, ct_hx, H_x.rows(), var->size());
      ct_hx += var->size();
    }

    // Append our residual and move forward
    res_big.block(ct_meas, 0, res.rows(), 1) = res;
    ct_meas += res.rows();
    it2++;
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Log which state variables are included in this update (for TinyVIO parity comparison)
  PRINT_DEBUG("[HX_ORDER] n_vars=%zu total_cols=%zu\n", Hx_order_big.size(), ct_jacob);
  for (size_t i = 0; i < Hx_order_big.size(); i++) {
    auto& var = Hx_order_big[i];
    PRINT_DEBUG("[HX_ORDER]   var[%zu]: id=%d size=%d\n", i, var->id(), var->size());
  }

  // We have appended all features to our Hx_big, res_big
  // Delete it so we do not reuse information
  for (size_t f = 0; f < feature_vec.size(); f++) {
    feature_vec[f]->to_delete = true;
  }

  // Return if we don't have anything and resize our matrices
  if (ct_meas < 1) {
    return;
  }
  assert(ct_meas <= max_meas_size);
  assert(ct_jacob <= max_hx_size);
  res_big.conservativeResize(ct_meas, 1);
  Hx_big.conservativeResize(ct_meas, ct_jacob);

  // DEBUG: Log actual matrix sizes after stacking features
  PRINT_DEBUG("[MATRIX_ACTUAL] n_features_stacked=%zu ct_meas=%zu ct_jacob=%zu\n",
              feature_vec.size(), ct_meas, ct_jacob);
  PRINT_DEBUG("[MATRIX_ACTUAL] Hx_big used: %zu rows x %zu cols (%.2f MB)\n",
              ct_meas, ct_jacob, (ct_meas * ct_jacob * 8.0) / (1024.0 * 1024.0));

  // 5. Perform measurement compression
  int rows_before_compress = (int)Hx_big.rows();
  int state_cols = (int)Hx_big.cols();

  // MSCKF_STACK: Stacked system before compression (for parity comparison)
  PRINT_DEBUG("[MSCKF_STACK] rows=%d cols=%d H_norm=%e res_norm=%e\n",
              rows_before_compress, state_cols, Hx_big.norm(), res_big.norm());

  UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
  PRINT_DEBUG("[COMPRESS] before=%d state_cols=%d after=%d\n",
              rows_before_compress, state_cols, (int)Hx_big.rows());

  // MSCKF_COMPRESS: After compression (for parity comparison)
  PRINT_DEBUG("[MSCKF_COMPRESS] rows=%d cols=%d H_norm=%e res_norm=%e\n",
              (int)Hx_big.rows(), (int)Hx_big.cols(), Hx_big.norm(), res_big.norm());
  if (Hx_big.rows() < 1) {
    return;
  }

  // Dump H and res after compression for parity comparison
  PRINT_DEBUG("[PARITY_H] rows=%d cols=%d\n", (int)Hx_big.rows(), (int)Hx_big.cols());
  for (int r = 0; r < (int)Hx_big.rows() && r < 6; ++r) {
    std::stringstream ss;
    ss << "[PARITY_H] H[" << r << "] = ";
    for (int c = 0; c < (int)Hx_big.cols(); ++c) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.9f ", Hx_big(r, c));
      ss << buf;
    }
    PRINT_DEBUG("%s\n", ss.str().c_str());
  }
  {
    std::stringstream ss;
    ss << "[PARITY_H] res = ";
    for (int r = 0; r < (int)res_big.rows() && r < 10; ++r) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.9f ", res_big(r));
      ss << buf;
    }
    PRINT_DEBUG("%s\n", ss.str().c_str());
  }

  rT4 = boost::posix_time::microsec_clock::local_time();

  // Our noise is isotropic, so make it here after our compression
  Eigen::MatrixXd R_big = _options.sigma_pix_sq * Eigen::MatrixXd::Identity(res_big.rows(), res_big.rows());

  // Noise logging for TinyVIO comparison
  PRINT_DEBUG("[NOISE] sigma_pix=%.6f sigma_pix_sq=%.9f R_rows=%d\n",
              _options.sigma_pix, _options.sigma_pix_sq, (int)R_big.rows());

  // 6. With all good features update the state
  StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
  // Comparison logging for TinyVIO parity verification
  PRINT_DEBUG("[MSCKF_UPD] feats_used=%d res_rows=%d\n", (int)feature_vec.size(), (int)res_big.rows());
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Debug print timing information
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to clean\n", (rT1 - rT0).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds to triangulate\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds create system (%d features)\n", (rT3 - rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds compress system\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[MSCKF-UP]: %.4f seconds update state (%d size)\n", (rT5 - rT4).total_microseconds() * 1e-6, (int)res_big.rows());
  PRINT_ALL("[MSCKF-UP]: %.4f seconds total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}
