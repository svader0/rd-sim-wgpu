@group(0) @binding(0) var texture_src: texture_2d<f32>;
@group(0) @binding(1) var texture_dst: texture_storage_2d<rg32float, write>;

struct PaintParams {
    center_x: f32,
    center_y: f32,
}

@group(0) @binding(2) var<uniform> paint_params: PaintParams;

// Paint brush radius in pixels
const BRUSH_RADIUS: f32 = 1.0;

// Take the current state of the simulation as a texture, 
// and add a "paint" of chemical B at the specified center.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let tex_size = textureDimensions(texture_dst);
    
    if (global_id.x >= tex_size.x || global_id.y >= tex_size.y) {
        return;
    }
    
    // Read current value
    let current = textureLoad(texture_src, pos, 0).rg;
    
    // Calculate distance from brush center
    let dx = f32(pos.x) - paint_params.center_x;
    let dy = f32(pos.y) - paint_params.center_y;
    let dist = sqrt(dx * dx + dy * dy);
    
    // Hard circular brush
    if (dist <= BRUSH_RADIUS) {
        textureStore(texture_dst, pos, vec4<f32>(current.r, 1.0, 0.0, 0.0));
    } else {
        textureStore(texture_dst, pos, vec4<f32>(current, 0.0, 0.0));
    }
}
