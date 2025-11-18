@group(0) @binding(0) var texture_src: texture_2d<f32>;
@group(0) @binding(1) var texture_dst: texture_storage_2d<rg32float, write>;

struct PaintParams {
    center_x: f32,
    center_y: f32,
}

@group(0) @binding(2) var<uniform> paint_params: PaintParams;

// Take the current state of the simulation as a texture, 
// and add a single "paint" of chemical B at the specified center.
// Then, return the updated texture.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let tex_size = textureDimensions(texture_dst);
    
    if (global_id.x >= tex_size.x || global_id.y >= tex_size.y) {
        return;
    }
    
    // Read current value
    let current = textureLoad(texture_src, pos, 0).rg;
    
    // If this is the clicked point, set chemical B to 1.0, otherwise copy existing value
    if (pos.x == i32(paint_params.center_x) && pos.y == i32(paint_params.center_y)) {
        textureStore(texture_dst, pos, vec4<f32>(1.0, 1.0, 0.0, 0.0));
    } else {
        textureStore(texture_dst, pos, vec4<f32>(current, 0.0, 0.0));
    }
}
