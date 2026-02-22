use bevy::prelude::*;
use jamtil::animation;
use jamtil::prelude::*;

fn main() {
    App::default()
        .add_plugins(DefaultPlugins)
        .add_plugins(JamtilPlugin)
        .add_systems(Startup, spawn)
        .run();
}

#[derive(Clone, Copy, Lerp, Component)]
struct DummyData(f32);

fn spawn(mut commands: Commands) {
    let end = 69.0;
    commands.spawn_sequence(
        seq![
            system(move || println!("  animating 0.0 -> {end}")),
            // Spawn the animation and wait for it to finish.
            animate(animations![(
                DummyData(0.0),
                AnimationTarget::entity(),
                Keyframe(DummyData(end)),
                Duration(1.0),
            )]),
            // Print the final value.
            system(|data: Single<&DummyData>| println!("  {}", data.0)),
        ]
        .on_start(|| println!("starting sequence"))
        .on_end((|| println!("ending sequence"), exit)),
    );
}
