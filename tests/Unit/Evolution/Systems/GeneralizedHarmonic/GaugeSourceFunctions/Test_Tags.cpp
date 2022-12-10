// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Tags.hpp"

namespace {
template <size_t Dim>
void test() {
  TestHelpers::db::test_compute_tag<
      GeneralizedHarmonic::gauges::Tags::GaugeAndDerivativeCompute<Dim>>(
      "Variables(GaugeH,SpacetimeDerivGaugeH)");

  // Use Harmonic gauge since then we don't need to set any values and we can
  // still check that the compute tag works correctly.

  auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::Lapse<>, gr::Tags::Shift<Dim>,
          gr::Tags::SpacetimeNormalOneForm<Dim>,
          gr::Tags::SpacetimeNormalVector<Dim>,
          gr::Tags::SqrtDetSpatialMetric<>, gr::Tags::InverseSpatialMetric<Dim>,
          gr::Tags::SpacetimeMetric<Dim>, GeneralizedHarmonic::Tags::Pi<Dim>,
          GeneralizedHarmonic::Tags::Phi<Dim>,
          ::Events::Tags::ObserverMesh<Dim>, ::Tags::Time,
          ::Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>,
          ::Events::Tags::ObserverInverseJacobian<Dim, Frame::ElementLogical,
                                                  Frame::Inertial>,
          GeneralizedHarmonic::gauges::Tags::GaugeCondition>,
      db::AddComputeTags<
          GeneralizedHarmonic::gauges::Tags::GaugeAndDerivativeCompute<Dim>>>(
      Scalar<DataVector>{}, tnsr::I<DataVector, Dim, Frame::Inertial>{},
      tnsr::a<DataVector, Dim, Frame::Inertial>{},
      tnsr::A<DataVector, Dim, Frame::Inertial>{}, Scalar<DataVector>{},
      tnsr::II<DataVector, Dim, Frame::Inertial>{},
      tnsr::aa<DataVector, Dim, Frame::Inertial>{},
      tnsr::aa<DataVector, Dim, Frame::Inertial>{},
      tnsr::iaa<DataVector, Dim, Frame::Inertial>{},
      Mesh<Dim>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, 1.3,
      tnsr::I<DataVector, Dim, Frame::Inertial>{},
      InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                      Frame::Inertial>{},
      std::unique_ptr<GeneralizedHarmonic::gauges::GaugeCondition>{
          std::make_unique<GeneralizedHarmonic::gauges::Harmonic>()});

  const size_t num_points =
      db::get<::Events::Tags::ObserverMesh<Dim>>(box).number_of_grid_points();

  CHECK(db::get<GeneralizedHarmonic::Tags::GaugeH<Dim, Frame::Inertial>>(box) ==
        tnsr::a<DataVector, Dim, Frame::Inertial>(num_points, 0.0));
  CHECK(db::get<GeneralizedHarmonic::Tags::SpacetimeDerivGaugeH<
            Dim, Frame::Inertial>>(box) ==
        tnsr::ab<DataVector, Dim, Frame::Inertial>(num_points, 0.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.Tags",
                  "[Evolution][Unit]") {
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::gauges::Tags::GaugeCondition>("GaugeCondition");

  test<1>();
  test<2>();
  test<3>();
}