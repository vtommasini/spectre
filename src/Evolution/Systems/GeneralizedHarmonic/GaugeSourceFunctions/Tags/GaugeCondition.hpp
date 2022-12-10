// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace GeneralizedHarmonic::OptionTags {
struct Group;
}  // namespace GeneralizedHarmonic::OptionTags
/// \endcond

namespace GeneralizedHarmonic::gauges {
namespace OptionTags {
struct GaugeCondition {
  using type = std::unique_ptr<gauges::GaugeCondition>;
  static constexpr Options::String help{"The gauge condition to impose."};
  using group = GeneralizedHarmonic::OptionTags::Group;
};
}  // namespace OptionTags

namespace Tags {
/// \brief The gauge condition to impose.
struct GaugeCondition : db::SimpleTag {
  using type = std::unique_ptr<gauges::GaugeCondition>;
  using option_tags =
      tmpl::list<GeneralizedHarmonic::gauges::OptionTags::GaugeCondition>;

  static constexpr bool pass_metavariables = false;
  static std::unique_ptr<gauges::GaugeCondition> create_from_options(
      const std::unique_ptr<gauges::GaugeCondition>& gauge_condition) {
    return gauge_condition->get_clone();
  }
};

/// \brief Gauge condition \f$H_a\f$ and its spacetime derivative
/// \f$\partial_b H_a\f$
template <size_t Dim>
struct GaugeAndDerivativeCompute
    : ::Tags::Variables<
          tmpl::list<::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>,
                     ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
                         Dim, Frame::Inertial>>>,
      db::ComputeTag {
  using base = ::Tags::Variables<tmpl::list<
      ::GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>,
      ::GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<Dim, Frame::Inertial>>>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      gr::Tags::Lapse<>, gr::Tags::Shift<Dim>,
      gr::Tags::SpacetimeNormalOneForm<Dim>,
      gr::Tags::SpacetimeNormalVector<Dim>, gr::Tags::SqrtDetSpatialMetric<>,
      gr::Tags::InverseSpatialMetric<Dim>, gr::Tags::SpacetimeMetric<Dim>,
      GeneralizedHarmonic::Tags::Pi<Dim>, GeneralizedHarmonic::Tags::Phi<Dim>,
      ::Events::Tags::ObserverMesh<Dim>, ::Tags::Time,
      ::Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>,
      ::Events::Tags::ObserverInverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>,
      Tags::GaugeCondition>;

  static void function(
      gsl::not_null<return_type*> gauge_and_deriv,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
      const tnsr::a<DataVector, Dim, Frame::Inertial>&
          spacetime_unit_normal_one_form,
      const tnsr::A<DataVector, Dim, Frame::Inertial>& spacetime_unit_normal,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
      const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
      const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
      const Mesh<Dim>& mesh, double time,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inverse_jacobian,
      const gauges::GaugeCondition& gauge_condition);
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic::gauges