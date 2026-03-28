#![allow(clippy::type_complexity)]

use crate::{animation::AnimationPlugin, sequence::SequencePlugin, tween::TweenPlugin};
use bevy::prelude::*;

pub mod animation;
pub mod sequence;
pub mod system;
pub mod tween;

pub mod prelude {
    pub use super::animation::*;
    pub use super::sequence::*;
    pub use super::system::*;
    pub use super::tween::*;
    pub use super::{
        JamtilPlugin, alt, animations, lerp_component, on_end, on_start, parallel, seq,
    };
}

pub struct JamtilPlugin;

impl Plugin for JamtilPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((SequencePlugin, AnimationPlugin, TweenPlugin));
    }
}
