#![allow(clippy::type_complexity)]

use crate::{animation::AnimationPlugin, sequence::SequencePlugin};
use bevy::prelude::*;

pub mod animation;
pub mod sequence;
pub mod system;

pub mod prelude {
    pub use super::JamtilPlugin;
    pub use super::animation::*;
    pub use super::sequence::*;
    pub use super::system::*;
    pub use super::{alt, on_end, on_start, seq};
    pub use super::{animations, parallel};
}

pub struct JamtilPlugin;

impl Plugin for JamtilPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((SequencePlugin, AnimationPlugin));
    }
}
