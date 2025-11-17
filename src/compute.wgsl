// Gray-Scott reaction-diffusion compute shader
// Uses ping-pong RG32Float textures (R channel=U, G channel=V)

struct SimParams {
    feed_rate: f32,      // F parameter
    kill_rate: f32,      // k parameter
    diffuse_u: f32,      // Du parameter
    diffuse_v: f32,      // Dv parameter
    delta_time: f32,     // dt parameter
    noise_strength: f32, // noise injection strength
    grid_width: u32,
    grid_height: u32,
    kernel_type: u32,    // 0=default, 1=cross, 2=diagonal, 3=spiral
    boundary_mode: u32,  // 0=wrap, 1=clamp, 2=reflect
    map_mode: u32,       // 0=off, 1=parameter map mode
    _padding: u32,
}

@group(0) @binding(0) var texture_src: texture_2d<f32>;
@group(0) @binding(1) var texture_dst: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var texture_sampler: sampler;
@group(0) @binding(3) var<uniform> params: SimParams;

// Boundary handling helper - returns wrapped, clamped, or reflected coordinates
fn apply_boundary(coord: i32, size: i32, mode: u32) -> i32 {
    if mode == 0u {
        // Wrap (toroidal)
        return (coord + size) % size;
    } else if mode == 1u {
        // Clamp (edges fixed)
        return clamp(coord, 0, size - 1);
    } else {
        // Reflect (mirror) - proper reflection that creates mirror image
        var c = coord;
        // Reflect negative coordinates
        if c < 0 {
            c = -c;
        }
        // Reflect coordinates beyond size
        if c >= size {
            c = 2 * (size - 1) - c;
        }
        // Handle double reflections
        if c < 0 {
            c = -c;
        }
        if c >= size {
            c = 2 * (size - 1) - c;
        }
        return clamp(c, 0, size - 1);
    }
}

// Random noise function using hash
fn hash(p: vec2<u32>) -> f32 {
    var h = p.x * 374761393u + p.y * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return f32(h) / 4294967295.0;
}

// Multiple Laplacian kernel implementations with configurable boundaries
fn laplacian(pos: vec2<i32>) -> vec2<f32> {
    let width = i32(params.grid_width);
    let height = i32(params.grid_height);
    
    let center = textureLoad(texture_src, pos, 0).rg;
    
    if params.kernel_type == 0u {
        // Default kernel (9-point stencil with diagonals):
        // [0.05, 0.2, 0.05]
        // [0.2, -1.0, 0.2]
        // [0.05, 0.2, 0.05]
        
        let left = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), pos.y), 0).rg;
        let right = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), pos.y), 0).rg;
        let up = textureLoad(texture_src, vec2<i32>(pos.x, apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg;
        let down = textureLoad(texture_src, vec2<i32>(pos.x, apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg;
        
        let up_left = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg;
        let up_right = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg;
        let down_left = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg;
        let down_right = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg;
        
        return left * 0.2 + right * 0.2 + up * 0.2 + down * 0.2 +
               up_left * 0.05 + up_right * 0.05 + down_left * 0.05 + down_right * 0.05 +
               center * -1.0;
               
    } else if params.kernel_type == 1u {
        // Cross kernel (only orthogonal neighbors, balanced):
        // Weights: 0.2 * 4 = 0.8, center = -0.8, total = 0
        // [  0,  0.2,   0]
        // [0.2, -0.8, 0.2]
        // [  0,  0.2,   0]
        
        let left = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), pos.y), 0).rg;
        let right = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), pos.y), 0).rg;
        let up = textureLoad(texture_src, vec2<i32>(pos.x, apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg;
        let down = textureLoad(texture_src, vec2<i32>(pos.x, apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg;
        
        return left * 0.2 + right * 0.2 + up * 0.2 + down * 0.2 + center * -0.8;
        
    } else if params.kernel_type == 2u {
        // Diagonal kernel (only diagonal neighbors, balanced):
        // Weights: 0.2 * 4 = 0.8, center = -0.8, total = 0
        // [0.2,   0, 0.2]
        // [  0,-0.8,   0]
        // [0.2,   0, 0.2]
        
        let up_left = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg;
        let up_right = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg;
        let down_left = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg;
        let down_right = textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg;
        
        return up_left * 0.2 + up_right * 0.2 + down_left * 0.2 + down_right * 0.2 + center * -0.8;
        
    } else if params.kernel_type == 3u {
        // Spiral kernel (5x5 symmetric, balanced to sum to zero):
        // Weights sum to zero: 4+2+6+8+10+10+8+6+2+4 = 60, center = -60
        
        var sum = vec2<f32>(0.0);
        
        // Row -2 (y-2)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 2, width, params.boundary_mode), apply_boundary(pos.y - 2, height, params.boundary_mode)), 0).rg * (4.0/60.0);
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), apply_boundary(pos.y - 2, height, params.boundary_mode)), 0).rg * (2.0/60.0);
        
        // Row -1 (y-1)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 2, width, params.boundary_mode), apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg * (6.0/60.0);
        
        // Row 0 (center)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 2, width, params.boundary_mode), pos.y), 0).rg * (8.0/60.0);
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), pos.y), 0).rg * (10.0/60.0);
        sum += center * (-60.0/60.0);
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), pos.y), 0).rg * (10.0/60.0);
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 2, width, params.boundary_mode), pos.y), 0).rg * (8.0/60.0);
        
        // Row +1 (y+1)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 2, width, params.boundary_mode), apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg * (6.0/60.0);
        
        // Row +2 (y+2)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), apply_boundary(pos.y + 2, height, params.boundary_mode)), 0).rg * (2.0/60.0);
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 2, width, params.boundary_mode), apply_boundary(pos.y + 2, height, params.boundary_mode)), 0).rg * (4.0/60.0);
        
        return sum;
    } else {
        // Custom asymmetric kernel (balanced to sum to zero):
        // Creates interesting directional patterns
        // [0.15, 0.10, 0.05]
        // [0.20, -0.80, 0.15]
        // [0.05, 0.05, 0.05]
        // Sum: (0.15+0.10+0.05+0.20+0.15+0.05+0.05+0.05) = 0.80, center = -0.80
        
        var sum = vec2<f32>(0.0);
        
        // Row -1 (y-1)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg * 0.15;
        sum += textureLoad(texture_src, vec2<i32>(pos.x, apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg * 0.10;
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), apply_boundary(pos.y - 1, height, params.boundary_mode)), 0).rg * 0.05;
        
        // Row 0 (center)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), pos.y), 0).rg * 0.20;
        sum += center * -0.80;
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), pos.y), 0).rg * 0.15;
        
        // Row +1 (y+1)
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x - 1, width, params.boundary_mode), apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg * 0.05;
        sum += textureLoad(texture_src, vec2<i32>(pos.x, apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg * 0.05;
        sum += textureLoad(texture_src, vec2<i32>(apply_boundary(pos.x + 1, width, params.boundary_mode), apply_boundary(pos.y + 1, height, params.boundary_mode)), 0).rg * 0.05;
        
        return sum;
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = vec2<i32>(global_id.xy);
    let dims = vec2<i32>(i32(params.grid_width), i32(params.grid_height));
    
    // Boundary check
    if (pos.x >= dims.x || pos.y >= dims.y) {
        return;
    }
    
    let uv = textureLoad(texture_src, pos, 0).rg;
    let u = uv.r;
    let v = uv.g;
    
    // Compute Laplacian
    let lap = laplacian(pos);
    let laplacian_u = lap.r;
    let laplacian_v = lap.g;
    
    // Calculate position-dependent parameters for map mode
    var feed = params.feed_rate;
    var kill = params.kill_rate;
    
    if params.map_mode != 0u {
        // Map mode: F varies along Y axis, k varies along X axis
        let norm_x = f32(pos.x) / f32(params.grid_width);
        let norm_y = f32(pos.y) / f32(params.grid_height);
        
        kill = mix(0.045, 0.070, norm_x);  // k: 0.045 to 0.070 along X
        feed = mix(0.010, 0.100, norm_y);  // F: 0.01 to 0.1 along Y
    }
    
    // Gray-Scott equations:
    // ∂u/∂t = Du·∇²u - u·v² + F·(1-u)
    // ∂v/∂t = Dv·∇²v + u·v² - (F+k)·v
    
    let uvv = u * v * v;
    
    let du_dt = params.diffuse_u * laplacian_u - uvv + feed * (1.0 - u);
    let dv_dt = params.diffuse_v * laplacian_v + uvv - (feed + kill) * v;
    
    // Forward Euler integration
    var new_u = u + du_dt * params.delta_time;
    var new_v = v + dv_dt * params.delta_time;
    
    // Add noise injection if enabled
    if params.noise_strength > 0.0 {
        let noise = hash(vec2<u32>(pos)) * 2.0 - 1.0; // Range: -1 to 1
        new_u += noise * params.noise_strength;
        new_v += noise * params.noise_strength * 0.5; // Less noise on V
    }
    
    // Clamp to valid range
    let result = vec2<f32>(clamp(new_u, 0.0, 1.0), clamp(new_v, 0.0, 1.0));
    
    textureStore(texture_dst, pos, vec4<f32>(result, 0.0, 1.0));
}
