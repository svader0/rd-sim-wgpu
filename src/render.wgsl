struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Full-screen triangle
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.tex_coords = vec2<f32>(x, y);
    
    return out;
}

/*
  ___________________________
< Shader?! I hardly know her! >
  ---------------------------
         \   ^__^ 
          \  (oo)\_______
             (__)\       )\/\\
                 ||----w |
                 ||     ||
*/

@group(0) @binding(0) var reaction_texture: texture_2d<f32>;


struct RenderParams {
    color_palette: u32,
    emboss_enabled: u32,
    boundary_mode: u32,
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
}

@group(0) @binding(2) var<uniform> render_params: RenderParams;

struct GradientStop {
    position: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    color: vec4<f32>,
}

struct GradientData {
    stops: array<GradientStop, 8>,
    num_stops: u32,
    _padding: vec3<u32>,
}

@group(0) @binding(3) var<uniform> gradient: GradientData;

// Sample color from gradient at position t (0.0 to 1.0)
fn sample_gradient(t: f32) -> vec3<f32> {
    let val = clamp(t, 0.0, 1.0);
    
    // Handle edge cases
    if gradient.num_stops == 0u {
        return vec3<f32>(1.0, 1.0, 1.0); // White fallback
    }
    if gradient.num_stops == 1u {
        return gradient.stops[0].color.rgb;
    }
    if val <= gradient.stops[0].position {
        return gradient.stops[0].color.rgb;
    }
    if val >= gradient.stops[gradient.num_stops - 1u].position {
        return gradient.stops[gradient.num_stops - 1u].color.rgb;
    }
    
    // Find the two stops to interpolate between
    for (var i = 0u; i < gradient.num_stops - 1u; i = i + 1u) {
        if val >= gradient.stops[i].position && val <= gradient.stops[i + 1u].position {
            let stop1 = gradient.stops[i];
            let stop2 = gradient.stops[i + 1u];
            
            let range = stop2.position - stop1.position;
            if range > 0.0 {
                let local_t = (val - stop1.position) / range;
                return mix(stop1.color.rgb, stop2.color.rgb, local_t);
            } else {
                return stop1.color.rgb;
            }
        }
    }
    
    // Fallback (should never reach here)
    return gradient.stops[0].color.rgb;
}

// Color mapping for V channel
fn value_to_color(v: f32) -> vec3<f32> {

    return sample_gradient(v);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply zoom and pan to texture coordinates
    let zoomed_coords = (in.tex_coords - vec2<f32>(0.5, 0.5)) / render_params.zoom + vec2<f32>(0.5, 0.5);
    let panned_coords = zoomed_coords + vec2<f32>(render_params.pan_x, render_params.pan_y);
    
    let tex_size = textureDimensions(reaction_texture);
    
    let pixel_coord = panned_coords * vec2<f32>(tex_size) - vec2<f32>(0.5, 0.5);
    let base_coord = floor(pixel_coord);
    let frac = pixel_coord - base_coord;
    
    var sum = 0.0;
    var weight_sum = 0.0;
    
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let coord = vec2<i32>(base_coord) + vec2<i32>(dx, dy);
            
            var final_coord: vec2<i32>;
            var out_of_bounds = false;
            
            // Apply boundary condition
            if render_params.boundary_mode == 0u {
                // Wrap (toroidal topology)
                final_coord = vec2<i32>(
                    ((coord.x % i32(tex_size.x)) + i32(tex_size.x)) % i32(tex_size.x),
                    ((coord.y % i32(tex_size.y)) + i32(tex_size.y)) % i32(tex_size.y)
                );
            } else {
                // Clamp or Reflect - show black outside bounds
                if coord.x < 0 || coord.x >= i32(tex_size.x) || coord.y < 0 || coord.y >= i32(tex_size.y) {
                    out_of_bounds = true;
                }
                final_coord = clamp(coord, vec2<i32>(0, 0), vec2<i32>(i32(tex_size.x) - 1, i32(tex_size.y) - 1));
            }
            
            let dist = vec2<f32>(f32(dx), f32(dy)) - (frac - vec2<f32>(0.5, 0.5));
            let weight = exp(-dot(dist, dist) * 2.0);
            
            var sample_val = 0.0;
            // If we're not out of bounds, sample the texture; otherwise, we just show black.
            if !out_of_bounds {
                sample_val = textureLoad(reaction_texture, final_coord, 0).g;
            }
            sum += sample_val * weight;
            weight_sum += weight;
        }
    }
    
    let v = sum / weight_sum;
    var color = value_to_color(v);
    
    /*
        EMBOSSING! --------------------

        This is where the magic happens. It's mostly pieced together from stuff I found online,
        but we're basically calculating the surface normal based on the color gradients, then using that
        normal to do some simple Phong lighting with a couple light sources to give it a 3D embossed look.

    */
    if render_params.emboss_enabled != 0u {
        // For non-wrap modes, skip emboss near edges to prevent artifacts
        var apply_emboss = true;
        if render_params.boundary_mode != 0u {
            let border = 2;
            if panned_coords.x < f32(border) / f32(tex_size.x) || 
               panned_coords.x > f32(i32(tex_size.x) - border) / f32(tex_size.x) ||
               panned_coords.y < f32(border) / f32(tex_size.y) || 
               panned_coords.y > f32(i32(tex_size.y) - border) / f32(tex_size.y) {
                apply_emboss = false;
            }
        }
        
        if apply_emboss {
            // Calculate surface normal from gradients
            let coord_center = vec2<i32>(base_coord);
            
            // Apply boundary mode to neighbor sampling
            var coord_right: vec2<i32>;
            var coord_left: vec2<i32>;
            var coord_up: vec2<i32>;
            var coord_down: vec2<i32>;
            
            if render_params.boundary_mode == 0u {
                // Wrap boundary
                let wrapped_center = vec2<i32>(
                    (coord_center.x + i32(tex_size.x)) % i32(tex_size.x),
                    (coord_center.y + i32(tex_size.y)) % i32(tex_size.y)
                );
                coord_right = vec2<i32>((wrapped_center.x + 1) % i32(tex_size.x), wrapped_center.y);
                coord_left = vec2<i32>((wrapped_center.x - 1 + i32(tex_size.x)) % i32(tex_size.x), wrapped_center.y);
                coord_up = vec2<i32>(wrapped_center.x, (wrapped_center.y + 1) % i32(tex_size.y));
                coord_down = vec2<i32>(wrapped_center.x, (wrapped_center.y - 1 + i32(tex_size.y)) % i32(tex_size.y));
            } else {
                // Clamp boundary for non-wrap modes
                coord_right = clamp(coord_center + vec2<i32>(1, 0), vec2<i32>(0, 0), vec2<i32>(i32(tex_size.x) - 1, i32(tex_size.y) - 1));
                coord_left = clamp(coord_center + vec2<i32>(-1, 0), vec2<i32>(0, 0), vec2<i32>(i32(tex_size.x) - 1, i32(tex_size.y) - 1));
                coord_up = clamp(coord_center + vec2<i32>(0, 1), vec2<i32>(0, 0), vec2<i32>(i32(tex_size.x) - 1, i32(tex_size.y) - 1));
                coord_down = clamp(coord_center + vec2<i32>(0, -1), vec2<i32>(0, 0), vec2<i32>(i32(tex_size.x) - 1, i32(tex_size.y) - 1));
            }
            
            let val_right = textureLoad(reaction_texture, coord_right, 0).g;
            let val_left = textureLoad(reaction_texture, coord_left, 0).g;
            let val_up = textureLoad(reaction_texture, coord_up, 0).g;
            let val_down = textureLoad(reaction_texture, coord_down, 0).g;
            
            // Calculate gradients with strong height amplification for liquid appearance
            let dx = (val_right - val_left) * 0.5;
            let dy = (val_up - val_down) * 0.5;
            
            // Create normal vector
            let normal = normalize(vec3<f32>(-dx * 10.0, -dy * 10.0, 1.0));
            
            // Light from top-right (main light)
            let key_light = normalize(vec3<f32>(0.6, 0.3, 1.0));
            let key_diffuse = max(dot(normal, key_light), 0.0);
            
            // Rim light from left-back for edge highlights (oooOOOooo!)
            let rim_light = normalize(vec3<f32>(-0.8, 0.2, 0.5));
            let rim_diffuse = max(dot(normal, rim_light), 0.0);
            
            // STRONG specular highlight for making it look like wet/shiny liquid
            let view_dir = vec3<f32>(0.0, 0.0, 1.0); // Camera looking straight down
            let reflect_dir = reflect(-key_light, normal);
            let specular = pow(max(dot(view_dir, reflect_dir), 0.0), 64.0); // High shininess
            
            // Combine lighting components for liquid-like appearance
            let ambient = 0.3;
            let diffuse = key_diffuse * 0.7 + rim_diffuse * 0.2;
            let lighting = ambient + diffuse + specular * 1.5; // Strong specular
            
            color = color * lighting;
        }
    }
    
    return vec4<f32>(color, 1.0);
}
