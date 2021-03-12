#include "ukf.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

double normalize_angle(double angle) {
  double x = fmod(angle + M_PI, 2 * M_PI);
  if (x < 0) {
    x += 2.0 * M_PI;
  }
  return x - M_PI;
}

// Compute the weighted sum of all the columns of a matrix
VectorXd columnWeightedSum(const MatrixXd& m, const VectorXd& w)
{
  return (m * w.asDiagonal()).rowwise().sum();
}

template<typename Func>
void applyFunctionToColumns(const MatrixXd& m, Func f, MatrixXd& out)
{
  for (int i = 0; i < m.cols(); ++i)
    out.col(i) = f(m.col(i));
}

}  // namespace

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Not yet initialized
  is_initialized_ = false;

  // Dimensions of the problem
  n_x_ = 5;
  n_aug_ = 7;
  n_sig_ = 2 * n_aug_ + 1;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  // Initialise the parameter lambda and the number of variables
  
  lambda_ = 3.0 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Initialiase the weights
  weights_ = VectorXd(n_sig_);
  const double wd = 1.0 / (lambda_ + n_aug_);
  weights_(0) = lambda_ * wd;
  weights_.tail(n_sig_-1).fill(0.5 * wd);

  // Initialize covariance matrices
  las_cov_ = (VectorXd(2) << std_laspx_ * std_laspx_, std_laspy_ * std_laspy_).finished().asDiagonal();
  rad_cov_ = (VectorXd(3) << std_radr_ * std_radr_, std_radphi_ * std_radphi_, std_radrd_ * std_radrd_).finished().asDiagonal();

  NIS_laser_ = NIS_radar_ = 0.0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
    return;

  if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    return;

  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_.head(2) = meas_package.raw_measurements_;

      P_.topLeftCorner(2, 2) = las_cov_;
    } else  // if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      const double rho = meas_package.raw_measurements_(0);
      const double yaw = meas_package.raw_measurements_(1);
      const double rhod = meas_package.raw_measurements_(2);

      x_.head(3) << rho * cos(yaw), rho * sin(yaw), rhod;

      P_.topLeftCorner(3, 3).diagonal() << rad_cov_(0, 0), rad_cov_(0, 0), rad_cov_(2, 2);
    }
  } else {
    // Compute the prediction step
    const double delta_t = (meas_package.timestamp_ - time_us_) * 1.e-6;
    time_us_ = meas_package.timestamp_;

    Prediction(delta_t);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
      UpdateLidar(meas_package);
    else  // if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
      UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  // Augment x
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;
  
  // Augment P
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  const MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  auto Xsig_aug = MatrixXd(n_aug_, n_sig_);
  const auto sL = sqrt(lambda_ + n_aug_) * L;
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) = sL.colwise() + x_aug;
  Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) = - (sL.colwise() - x_aug);

  // Compute prediction sigma points
  {
    const auto p_x = Xsig_aug.row(0);
    const auto p_y = Xsig_aug.row(1);
    const auto v = Xsig_aug.row(2);
    const auto yaw = Xsig_aug.row(3);
    const auto yawd = Xsig_aug.row(4);
    const auto nu_a = Xsig_aug.row(5);
    const auto nu_yawdd = Xsig_aug.row(6);

    const auto sinYaw = yaw.unaryExpr(std::ptr_fun(sin));
    const auto cosYaw = yaw.unaryExpr(std::ptr_fun(cos));

    const auto yaw_p = yaw + delta_t * yawd;
    const auto sinYawP = yaw_p.unaryExpr(std::ptr_fun(sin));
    const auto cosYawP = yaw_p.unaryExpr(std::ptr_fun(cos));
    
    const auto v_over_yawd = v.array() / yawd.array();
    const double half_delta_square = 0.5 * delta_t * delta_t;
    const auto p_nu_a = half_delta_square * nu_a;

    const VectorXd p_x_p = p_x + 
      (yawd.cwiseAbs().array() > 1e-5).select(
        v_over_yawd * (sinYawP - sinYaw).array(),
        delta_t * v.cwiseProduct(cosYaw).array()).matrix()
      + p_nu_a.cwiseProduct(cosYaw);
    const VectorXd p_y_p = p_y + 
      (yawd.cwiseAbs().array() > 1e-5).select(
        v_over_yawd * (cosYaw - cosYawP).array(),
        delta_t * v.cwiseProduct(sinYaw).array()).matrix()
      + p_nu_a.cwiseProduct(sinYaw);
  
    Xsig_pred_.row(0) = p_x_p;
    Xsig_pred_.row(1) = p_y_p;
    Xsig_pred_.row(2) = v + delta_t * nu_a;
    Xsig_pred_.row(3) = yaw_p + half_delta_square * nu_yawdd;
    Xsig_pred_.row(4) = yawd + delta_t * nu_yawdd;
  }

  // Update state
  x_ = columnWeightedSum(Xsig_pred_, weights_);
  
  // Compute difference vector with angle normalization in [-pi, pi]
  MatrixXd x_diff = Xsig_pred_.colwise() - x_;
  x_diff.row(3) = x_diff.row(3).unaryExpr(std::ptr_fun(normalize_angle));

  // Update covariance matrix
  P_ = x_diff * weights_.asDiagonal() * x_diff.transpose();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Compute predicted lidar sigma points
  auto z_pred = x_.head(2);
  auto y = meas_package.raw_measurements_ - z_pred;

  auto S = P_.topLeftCorner(2, 2) + las_cov_;

  // Kalman Gain
  auto K = P_.leftCols(2) * S.inverse();

  // update state mean and covariance matrix
  x_ = x_ + K * y;
  P_ -= K * P_.topRows(2);

  NIS_laser_ = y.transpose() * S.inverse() * y;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Compute predicted radar sigma points
  const int n_z = 3;
  auto Zsig = MatrixXd(n_z, n_sig_);

  // applyFunctionToColumns(Xsig_pred_, [](auto col){
  //   const auto p_x = col(0);
  //   const auto p_y = col(1);
  //   const auto v = col(2);
  //   const auto yaw = col(3);

  //   const auto vx = v * cos(yaw);
  //   const auto vy = v * sin(yaw);

  //   VectorXd result(3);
  //   result(0) = sqrt(p_x * p_x + p_y * p_y);
  //   result(1) = atan2(p_y, p_x);
  //   result(2) = (p_x * vx + p_y * vy) / result(0);
  //   return result;
  // }, Zsig);

  const auto p_x = Xsig_pred_.row(0);
  const auto p_y = Xsig_pred_.row(1);
  const auto v = Xsig_pred_.row(2);
  const auto yaw = Xsig_pred_.row(3);

  const auto vx = v.array() * yaw.unaryExpr(std::ptr_fun(cos)).array();
  const auto vy = v.array() * yaw.unaryExpr(std::ptr_fun(sin)).array();

  Zsig.row(0) = (p_x.cwiseProduct(p_x) + p_y.cwiseProduct(p_y)).cwiseSqrt();
  Zsig.row(1) = p_y.binaryExpr(p_x, std::ptr_fun(atan2));
  Zsig.row(2) = (p_x.array() * vx + p_y.array() * vy) / Zsig.row(0).array();

  // Compute mean and covariance from the sigma points
  const VectorXd z_pred = columnWeightedSum(Zsig, weights_);

  MatrixXd z_diff = Zsig.colwise() - z_pred;
  z_diff.row(1) = z_diff.row(1).unaryExpr(std::ptr_fun(normalize_angle));

  MatrixXd S = z_diff * weights_.asDiagonal() * z_diff.transpose() + rad_cov_;

  // Compute cross-correlation

  MatrixXd x_diff = Xsig_pred_.colwise() - x_;
  x_diff.row(3) = x_diff.row(3).unaryExpr(std::ptr_fun(normalize_angle));

  auto Tc = x_diff * weights_.asDiagonal() * z_diff.transpose();

  // Kalman Gain
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  y(1) = normalize_angle(y(1));

  // update state mean and covariance matrix
  x_ = x_ + K * y;
  P_ -= K * S * K.transpose();

  NIS_radar_ = y.transpose() * S.inverse() * y;
}
