use crate::prelude::{AnimationSystems, Lerp};
use bevy::prelude::*;

pub struct TweenPlugin;

impl Plugin for TweenPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (SpriteColor::system, TransformQuat::system).in_set(TweenSystems),
        )
        .configure_sets(
            Update,
            TweenSystems
                .after(AnimationSystems::Interpolate)
                .before(AnimationSystems::Step),
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SystemSet)]
pub struct TweenSystems;

#[macro_export]
macro_rules! lerp_component {
    ($target:ty, $lerp:ident, $inner:ty, $($mutate:tt)*) => {
        #[derive(Clone, Copy, Lerp, Component)]
        pub struct $lerp(pub $inner);
        impl $lerp {
            fn system(mut components: Query<(&mut $target, &$lerp), Changed<$lerp>>) {
                let mutate = $($mutate)*;
                for (mut entity, component) in components.iter_mut() {
                    mutate(&mut entity, &component);
                }
            }
        }
    };
}

lerp_component!(
    Sprite,
    SpriteColor,
    Color,
    |e: &mut Sprite, c: &SpriteColor| { e.color = c.0 }
);

impl Lerp for Quat {
    fn lerp(&self, rhs: &Self, t: f32) -> Self {
        Quat::lerp(*self, *rhs, t)
    }
    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }
}

lerp_component!(
    Transform,
    TransformQuat,
    Quat,
    |e: &mut Transform, c: &TransformQuat| { e.rotation = c.0 }
);
