use super::*;
use bevy::ecs::system::BoxedSystem;
use std::marker::PhantomData;

pub fn exit(mut writer: MessageWriter<AppExit>) {
    writer.write(AppExit::Success);
}

pub fn exit_on_input<T: Clone + Eq + std::hash::Hash + Send + Sync + 'static>(
    input: T,
) -> impl FnMut(Res<ButtonInput<T>>, MessageWriter<AppExit>) {
    move |button: Res<ButtonInput<T>>, mut writer: MessageWriter<AppExit>| {
        if button.just_pressed(input.clone()) {
            writer.write(AppExit::Success);
        }
    }
}

pub fn system<S, M>(system: S) -> JamtilSystem
where
    S: IntoJamtilSystem<M>,
{
    system.into_jamtil_system()
}

#[derive(Component)]
pub struct JamtilSystem {
    pub(crate) system: Option<SystemInner>,
    pub(crate) initialized: bool,
}

impl JamtilSystem {
    fn new(inner: SystemInner) -> Self {
        Self {
            system: Some(inner),
            initialized: false,
        }
    }
}

pub(crate) enum SystemInner {
    Void(BoxedSystem),
    Entity(BoxedSystem<In<Entity>>),
    VoidCondition(BoxedSystem<(), bool>),
    EntityCondition(BoxedSystem<In<Entity>, bool>),
}

pub trait IntoJamtilSystem<M> {
    fn into_jamtil_system(self) -> JamtilSystem;
}

struct VoidSystem<T>(PhantomData<T>);
impl<S, M> IntoJamtilSystem<VoidSystem<M>> for S
where
    S: IntoSystem<(), (), M>,
{
    fn into_jamtil_system(self) -> JamtilSystem {
        JamtilSystem::new(SystemInner::Void(Box::new(IntoSystem::into_system(self))))
    }
}

struct EntitySystem<T>(PhantomData<T>);
impl<S, M> IntoJamtilSystem<EntitySystem<M>> for S
where
    S: IntoSystem<In<Entity>, (), M>,
{
    fn into_jamtil_system(self) -> JamtilSystem {
        JamtilSystem::new(SystemInner::Entity(Box::new(IntoSystem::into_system(self))))
    }
}

struct VoidCondition<T>(PhantomData<T>);
impl<S, M> IntoJamtilSystem<VoidCondition<M>> for S
where
    S: IntoSystem<(), bool, M>,
{
    fn into_jamtil_system(self) -> JamtilSystem {
        JamtilSystem::new(SystemInner::VoidCondition(Box::new(
            IntoSystem::into_system(self),
        )))
    }
}

struct EntityCondition<T>(PhantomData<T>);
impl<S, M> IntoJamtilSystem<EntityCondition<M>> for S
where
    S: IntoSystem<In<Entity>, bool, M>,
{
    fn into_jamtil_system(self) -> JamtilSystem {
        JamtilSystem::new(SystemInner::EntityCondition(Box::new(
            IntoSystem::into_system(self),
        )))
    }
}
