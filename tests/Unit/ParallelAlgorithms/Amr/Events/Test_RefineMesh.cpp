// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Amr/Criteria/IncreaseResolution.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Events/RefineMesh.hpp"
#include "ParallelAlgorithms/Amr/Projectors/CopyFromCreatorOrLeaveAsIs.hpp"
#include "ParallelAlgorithms/Amr/Projectors/DefaultInitialize.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

template <typename Metavariables>
struct ElementComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;
  using const_global_cache_tags =
      tmpl::list<amr::Criteria::Tags::Criteria,
                 logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>;
  using simple_tags =
      tmpl::list<domain::Tags::Element<1>, domain::Tags::Mesh<1>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 1;
  using component_list = tmpl::list<ElementComponent<Metavariables>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<amr::Events::RefineMesh>>,
                  tmpl::pair<amr::Criterion,
                             tmpl::list<amr::Criteria::IncreaseResolution<1>>>>;
  };

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors =
        tmpl::list<::amr::projectors::DefaultInitialize<
                       Parallel::Tags::GlobalCacheImpl<Metavariables>>,
                   ::amr::projectors::CopyFromCreatorOrLeaveAsIs<
                       ::domain::Tags::Element<1>>>;
  };
};

void test(const Event& event) {
  CHECK(event.needs_evolved_variables());
  std::vector<std::unique_ptr<amr::Criterion>> criteria;
  criteria.emplace_back(
      std::make_unique<amr::Criteria::IncreaseResolution<1>>());
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {std::move(criteria), ::Verbosity::Debug}};
  using element_component = ElementComponent<Metavariables>;
  const ElementId<1> element_id{0};
  const Element<1> element{element_id, DirectionMap<1, Neighbors<1>>{}};
  const Mesh<1> mesh{std::array{3_st}, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  ActionTesting::emplace_component_and_initialize<element_component>(
      &runner, element_id, {element, mesh});
  auto& box = ActionTesting::get_databox<element_component>(
      make_not_null(&runner), element_id);
  auto obs_box = make_observation_box<tmpl::list<>>(make_not_null(&box));
  event.run(make_not_null(&obs_box),
            ActionTesting::cache<element_component>(runner, element_id),
            element_id, std::add_pointer_t<element_component>{},
            {"Unused", -1.0});
  const Mesh<1> expected_mesh{std::array{4_st}, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  CHECK(
      ActionTesting::get_databox_tag<element_component, domain::Tags::Mesh<1>>(
          runner, element_id) == expected_mesh);
  CHECK(ActionTesting::get_databox_tag<element_component,
                                       domain::Tags::Element<1>>(
            runner, element_id) == element);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Events.RefineMesh", "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();
  const amr::Events::RefineMesh event{};
  test(event);
  test(serialize_and_deserialize(event));
  const auto option_event =
      TestHelpers::test_creation<std::unique_ptr<Event>, Metavariables>(
          "RefineMesh\n");
  test(*option_event);
  test(*serialize_and_deserialize(option_event));
}