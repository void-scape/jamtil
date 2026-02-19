#![allow(clippy::type_complexity)]

use crate::sequence::SequencePlugin;
use bevy::prelude::*;

mod sequence;
mod system;

pub mod prelude {
    pub use super::JamtilPlugin;
    pub use super::sequence::*;
    pub use super::system::*;
    pub use super::{alt, on_end, on_start, seq};
}

pub struct JamtilPlugin;

impl Plugin for JamtilPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SequencePlugin);
    }
}
