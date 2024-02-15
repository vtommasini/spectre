// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

struct Packet {
  // Constructor
  Packet(const double& time_, const double& coord_x_, const double& coord_y_,
         const double& coord_z_, const double& p_upper_t_, const double& p_x_,
         const double& p_y_, const double& p_z_)
      : time(time_), momentum_upper_t(p_upper_t_) {
    coordinates[0] = coord_x_;
    coordinates[1] = coord_y_;
    coordinates[2] = coord_z_;
    momentum[0] = p_x_;
    momentum[1] = p_y_;
    momentum[2] = p_z_;
  }

  // Current time
  double time;

  // p^t
  double momentum_upper_t;

  // Coordinates of the packet, currently in Inertial coordinates
  tnsr::I<double, 3, Frame::ElementLogical> coordinates;

  // Spatial components of the 4-momentum, also in Inertial coordinates
  tnsr::i<double, 3, Frame::Inertial> momentum;

  // Recalculte p^t using the fact that the 4-momentum is a null vector
  void renormalize_momentum(
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& lapse, const size_t& closest_point_index);
};

}  // namespace Particles::MonteCarlo
