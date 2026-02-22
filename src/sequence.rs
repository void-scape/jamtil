// TODO: on_reset

use crate::system::*;
use bevy::prelude::*;
use std::marker::PhantomData;

pub struct SequencePlugin;

impl Plugin for SequencePlugin {
    fn build(&self, app: &mut App) {
        app.add_observer(step_sequence)
            .add_observer(end_sequence_child)
            .add_observer(sequence::reset)
            .add_observer(sequence::eval)
            //
            .add_observer(step_alt)
            .add_observer(end_alt_child)
            .add_observer(alt::reset)
            .add_observer(alt::eval)
            //
            .add_observer(step_systems)
            //
            .add_observer(eval_limit)
            .add_observer(step_repeat)
            .add_observer(reset)
            //
            .add_systems(PreUpdate, poll_system.in_set(SequenceSystems));
        //
        // .add_observer(debug::<Reset, ()>)
        // .add_observer(debug::<Evaluate, ()>)
        // .add_observer(debug::<Insert, Start>)
        // .add_observer(debug::<StepStart, ()>)
        // .add_observer(debug::<StepEnd, ()>)
        // .add_observer(debug::<Insert, End>)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SystemSet)]
pub struct SequenceSystems;

#[derive(Default, Component)]
pub enum Evaluation {
    #[default]
    True,
    False,
}

impl Evaluation {
    pub fn is_true(&self) -> bool {
        matches!(self, Self::True)
    }

    pub fn is_false(&self) -> bool {
        !self.is_true()
    }
}

// TODO: make this propagate on insertion
#[derive(Default, Component)]
pub struct RequiresEvaluation;

#[derive(EntityEvent)]
pub struct Reset(Entity);

fn reset(reset: On<Reset>, mut commands: Commands) {
    commands.entity(reset.0).remove::<(Start, End)>();
}

#[derive(EntityEvent)]
pub struct Evaluate {
    entity: Entity,
    kind: EvaluateKind,
}

impl Evaluate {
    pub fn shallow(entity: Entity) -> Self {
        Self {
            entity,
            kind: EvaluateKind::Shallow,
        }
    }

    pub fn deep(entity: Entity) -> Self {
        Self {
            entity,
            kind: EvaluateKind::Deep,
        }
    }
}

pub enum EvaluateKind {
    Shallow,
    Deep,
}

pub struct ShallowEvaluateEntity(pub Entity);
impl Command for ShallowEvaluateEntity {
    fn apply(self, world: &mut World) {
        // println!("[DEBUG] EvaluateEntity {}", self.0);
        world.entity_mut(self.0).trigger(Evaluate::shallow);
        world.flush();
    }
}

pub struct DeepEvaluateEntity(pub Entity);
impl Command for DeepEvaluateEntity {
    fn apply(self, world: &mut World) {
        // println!("[DEBUG] EvaluateEntity {}", self.0);
        world.entity_mut(self.0).trigger(Evaluate::deep);
        world.flush();
    }
}

#[derive(Component)]
pub struct Start;

pub struct StartEntity(pub Entity);
impl Command for StartEntity {
    fn apply(self, world: &mut World) {
        // println!("[DEBUG] StartEntity {}", self.0);
        let systems: Vec<Entity> = world
            .get::<StartSystems>(self.0)
            .map(|s| s.iter().collect())
            .unwrap_or_default();
        for system in systems {
            RunSystem {
                system,
                root: self.0,
                node: false,
            }
            .apply(world);
        }
        world.entity_mut(self.0).insert(Start);
        world.flush();
    }
}

#[derive(Component)]
struct End;

pub struct EndEntity(Entity);
impl Command for EndEntity {
    fn apply(self, world: &mut World) {
        // println!("[DEBUG] EndEntity {}", self.0);
        let systems: Vec<Entity> = world
            .get::<EndSystems>(self.0)
            .map(|s| s.iter().collect())
            .unwrap_or_default();
        for system in systems {
            RunSystem {
                system,
                root: self.0,
                node: false,
            }
            .apply(world);
        }
        world.entity_mut(self.0).insert(End);
        world.flush();
    }
}

#[derive(EntityEvent)]
pub struct StepStart(pub Entity);

#[derive(EntityEvent)]
pub struct StepStartEntity(pub Entity);
impl Command for StepStartEntity {
    fn apply(self, world: &mut World) {
        // println!("[DEBUG] StepStartEntity {}", self.0);
        if world.entity(self.0).contains::<RequiresEvaluation>() {
            DeepEvaluateEntity(self.0).apply(world);
        }
        world.entity_mut(self.0).trigger(StepStart);
        world.flush();
    }
}

#[derive(EntityEvent)]
pub struct StepEnd(pub Entity);

#[derive(EntityEvent)]
pub struct StepEndEntity(pub Entity);
impl Command for StepEndEntity {
    fn apply(self, world: &mut World) {
        // println!("[DEBUG] StepEndEntity {}", self.0);
        if world.entity(self.0).contains::<RequiresEvaluation>() {
            DeepEvaluateEntity(self.0).apply(world);
        }
        world.entity_mut(self.0).trigger(StepEnd);
        world.flush();
    }
}

macro_rules! related_node {
    ($mod:ident, $parent:ident, $child:ident) => {
        #[derive(Component)]
        #[relationship_target(relationship = $child, linked_spawn)]
        #[require(Evaluation, RequiresEvaluation)]
        pub struct $parent(Vec<Entity>);
        #[derive(Component)]
        #[relationship(relationship_target = $parent)]
        #[require(Evaluation)]
        pub struct $child(pub Entity);
        mod $mod {
            use super::*;
            pub fn reset(reset: On<Reset>, mut commands: Commands, parents: Query<&$parent>) {
                if let Ok(parents) = parents.get(reset.0) {
                    for entity in parents.iter() {
                        commands.entity(entity).trigger(Reset);
                    }
                }
            }
            pub fn eval(eval: On<Evaluate>, mut commands: Commands, parents: Query<&$parent>) {
                if let Ok(parents) = parents.get(eval.entity) {
                    if matches!(eval.kind, EvaluateKind::Deep) {
                        for entity in parents.iter() {
                            commands.queue(ShallowEvaluateEntity(entity));
                        }
                    }
                }
            }
        }
    };
}

related_node!(sequence, Sequence, SequenceOf);
#[macro_export]
macro_rules! seq {
    [$($c:expr),*$(,)?] => {
        ::bevy::ecs::related!($crate::sequence::Sequence [$($c),*])
    };
}

fn step_sequence(
    step: On<StepStart>,
    mut commands: Commands,
    parents: Query<(&Sequence, &Evaluation)>,
    nodes: Query<(Entity, Has<Start>, &Evaluation), Without<End>>,
) {
    if let Ok((children, eval)) = parents.get(step.0) {
        if eval.is_false() {
            commands.queue(StepEndEntity(step.0));
            return;
        }
        if let Some((next, has_start)) = nodes
            .iter_many(children.iter())
            .find_map(|(e, s, eval)| eval.is_true().then_some((e, s)))
        {
            if !has_start {
                commands.queue(StartEntity(next));
            }
            commands.queue(StepStartEntity(next));
        } else {
            commands.queue(StepEndEntity(step.0));
        }
    }
}

fn end_sequence_child(
    complete: On<Insert, End>,
    mut commands: Commands,
    parents: Query<&SequenceOf>,
) {
    if let Ok(parent) = parents.get(complete.entity) {
        commands.queue(StepStartEntity(parent.0));
    }
}

related_node!(alt, Alt, AltOf);
#[macro_export]
macro_rules! alt {
    [$($c:expr),*$(,)?] => {
        ::bevy::ecs::related!($crate::sequence::Alt [$($c),*])
    };
}

fn step_alt(
    step: On<StepStart>,
    mut commands: Commands,
    alts: Query<(&Alt, &Evaluation)>,
    children: Query<(Entity, Has<Start>, &Evaluation)>,
) {
    if let Ok((alts, eval)) = alts.get(step.0) {
        if eval.is_false() {
            commands.queue(StepEndEntity(step.0));
            return;
        }
        if let Some((next, has_start)) = children
            .iter_many(alts.iter())
            .find_map(|(e, s, eval)| eval.is_true().then_some((e, s)))
        {
            if !has_start {
                commands.queue(StartEntity(next));
            }
            commands.queue(StepStartEntity(next));
        } else {
            commands.queue(StepEndEntity(step.0));
        }
    }
}

fn end_alt_child(complete: On<Insert, End>, mut commands: Commands, parents: Query<&AltOf>) {
    if let Ok(parent) = parents.get(complete.entity) {
        commands.queue(StepEndEntity(parent.0));
    }
}

#[derive(Default, Component)]
pub struct Iterations(pub usize);

#[derive(Component)]
#[require(Evaluation, RequiresEvaluation, Iterations)]
pub enum Repeat {
    Loop,
    Times(usize),
}

fn step_repeat(
    step: On<StepEnd>,
    mut commands: Commands,
    mut repeat: Query<(Option<&Repeat>, &mut Iterations, &Evaluation)>,
) {
    if let Ok((repeat, mut iteration, eval)) = repeat.get_mut(step.0) {
        iteration.0 += 1;
        if let Some(repeat) = repeat {
            match repeat {
                Repeat::Loop => {}
                Repeat::Times(i) => {
                    if *i <= iteration.0 {
                        commands.queue(EndEntity(step.0));
                        return;
                    }
                }
            }
            if eval.is_false() {
                commands.queue(EndEntity(step.0));
            } else {
                commands.entity(step.0).trigger(Reset).insert(Start);
                commands.queue(StepStartEntity(step.0));
            }
        } else {
            commands.queue(EndEntity(step.0));
        }
    } else {
        commands.queue(EndEntity(step.0));
    }
}

#[derive(Component)]
#[require(Evaluation, RequiresEvaluation, Iterations)]
pub struct Limit(pub usize);

fn eval_limit(evaluate: On<Evaluate>, mut limits: Query<(&Limit, &Iterations, &mut Evaluation)>) {
    if let Ok((limit, iteration, mut eval)) = limits.get_mut(evaluate.entity)
        && limit.0 <= iteration.0
    {
        *eval = Evaluation::False;
    }
}

// -------
// Systems
// -------

struct RunSystem {
    system: Entity,
    root: Entity,
    node: bool,
}
// TODO: return a result
impl Command for RunSystem {
    /// Implementation adapted from [`World::run_system_with`].
    fn apply(self, world: &mut World) {
        let Ok(mut entity) = world.get_entity_mut(self.system) else {
            return;
        };
        let Some(mut registered_system) = entity.get_mut::<JamtilSystem>() else {
            return;
        };
        let mut system = registered_system.system.take().unwrap();
        if !registered_system.initialized {
            match &mut system {
                SystemInner::Void(system) => {
                    system.initialize(world);
                }
                SystemInner::Entity(system) => {
                    system.initialize(world);
                }
                SystemInner::VoidCondition(condition) => {
                    condition.initialize(world);
                }
                SystemInner::EntityCondition(condition) => {
                    condition.initialize(world);
                }
            }
        }
        // Wait to run the commands until the system is available again.
        // This is needed so the systems can recursively run themselves.
        let result = match &mut system {
            SystemInner::Void(system) => {
                _ = system.run_without_applying_deferred((), world);
                system.queue_deferred(world.into());
                true
            }
            SystemInner::Entity(system) => {
                _ = system.run_without_applying_deferred(self.root, world);
                system.queue_deferred(world.into());
                true
            }
            SystemInner::VoidCondition(condition) => {
                let result = condition.run_without_applying_deferred((), world).unwrap();
                condition.queue_deferred(world.into());
                result
            }
            SystemInner::EntityCondition(condition) => {
                let result = condition
                    .run_without_applying_deferred(self.root, world)
                    .unwrap();
                condition.queue_deferred(world.into());
                result
            }
        };
        // Return ownership of system trait object (if entity still exists)
        if let Ok(mut entity) = world.get_entity_mut(self.system)
            && let Some(mut registered_system) = entity.get_mut::<JamtilSystem>()
        {
            registered_system.system = Some(system);
            registered_system.initialized = true;
        }
        // Run any commands enqueued by the system
        world.flush();
        if self.node && result {
            world.entity_mut(self.system).remove::<PollSystem>();
            StepEndEntity(self.system).apply(world);
        }
    }
}

pub(super) fn step_systems(
    step: On<StepStart>,
    mut commands: Commands,
    systems: Query<(), With<JamtilSystem>>,
) {
    if systems.contains(step.0) {
        commands.entity(step.0).insert(PollSystem);
        commands.queue(RunSystem {
            root: step.0,
            system: step.0,
            node: true,
        });
    }
}

#[derive(Component)]
struct PollSystem;

fn poll_system(
    mut commands: Commands,
    systems: Query<Entity, (With<JamtilSystem>, With<PollSystem>)>,
) {
    for entity in systems.iter() {
        commands.queue(RunSystem {
            root: entity,
            system: entity,
            node: true,
        });
    }
}

#[macro_export]
macro_rules! on_start {
    [$($child:expr),*$(,)?] => {
        ::bevy::ecs::related!($crate::sequence::StartSystems [$($crate::system::system($child)),*])
    };
}

#[derive(Component)]
#[relationship_target(relationship = StartSystemOf, linked_spawn)]
#[require(Evaluation)]
pub struct StartSystems(Vec<Entity>);

#[derive(Component)]
#[relationship(relationship_target = StartSystems)]
pub struct StartSystemOf(pub Entity);

#[macro_export]
macro_rules! on_end {
    [$($child:expr),*$(,)?] => {
        ::bevy::ecs::related!($crate::sequence::EndSystems [$($crate::system::system($child)),*])
    };
}

#[derive(Component)]
#[relationship_target(relationship = EndSystemOf, linked_spawn)]
#[require(Evaluation)]
pub struct EndSystems(Vec<Entity>);

#[derive(Component)]
#[relationship(relationship_target = EndSystems)]
pub struct EndSystemOf(pub Entity);

// ----------------
// Extension traits
// ----------------

pub trait CommandsExt {
    fn spawn_sequence<T: Bundle>(&mut self, bundle: T) -> EntityCommands<'_>;
}

impl CommandsExt for Commands<'_, '_> {
    fn spawn_sequence<T: Bundle>(&mut self, bundle: T) -> EntityCommands<'_> {
        let entity = self.spawn(bundle).id();
        self.queue(StartEntity(entity));
        self.queue(StepStartEntity(entity));
        self.entity(entity)
    }
}

impl<T> SeqExt for T where T: Bundle {}
pub trait SeqExt: Sized {
    fn repeating(self) -> (Self, Repeat) {
        (self, Repeat::Loop)
    }

    fn repeated(self, times: usize) -> (Self, Repeat) {
        (self, Repeat::Times(times))
    }

    fn limit(self, iterations: usize) -> (Self, Limit) {
        (self, Limit(iterations))
    }

    fn on_start<M>(self, systems: impl SystemBundle<M>) -> (Self, impl Bundle) {
        (self, systems.into_start_bundle())
    }

    fn on_end<M>(self, systems: impl SystemBundle<M>) -> (Self, impl Bundle) {
        (self, systems.into_end_bundle())
    }
}

pub trait SystemBundle<M> {
    fn into_start_bundle(self) -> impl Bundle;
    fn into_end_bundle(self) -> impl Bundle;
}

#[doc(hidden)]
pub struct SingleSystem<M>(PhantomData<M>);
impl<T, M> SystemBundle<SingleSystem<M>> for T
where
    T: IntoJamtilSystem<M>,
{
    fn into_start_bundle(self) -> impl Bundle {
        on_start![self]
    }
    fn into_end_bundle(self) -> impl Bundle {
        on_end![self]
    }
}

macro_rules! impl_system_bundle {
    ($(($P:ident, $p:ident)),*) => {
        #[allow(non_camel_case_types)]
        impl<$($P, $p,)*> SystemBundle<($($p,)*)> for ($($P,)*)
        where
            $($P: IntoJamtilSystem<$p>,)*
        {
            fn into_start_bundle(self) -> impl Bundle {
                let ($($p,)*) = self;
                on_start![$($p,)*]
            }
            fn into_end_bundle(self) -> impl Bundle {
                let ($($p,)*) = self;
                on_end![$($p,)*]
            }
        }
    };
}
variadics_please::all_tuples!(impl_system_bundle, 1, 10, P, p);
