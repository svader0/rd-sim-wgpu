use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;
use std::cell::RefCell;
use std::rc::Rc;

const GRID_WIDTH: u32 = 2048;
const GRID_HEIGHT: u32 = 2048;
const MAX_GRADIENT_STOPS: usize = 8;

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GradientStop {
    position: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    color: [f32; 4],  // RGBA
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GradientData {
    stops: [GradientStop; MAX_GRADIENT_STOPS],
    num_stops: u32,
    _padding: [u32; 3],
    _final_padding: [f32; 4],  // Extra padding to match WGSL alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    feed_rate: f32,
    kill_rate: f32,
    diffuse_u: f32,
    diffuse_v: f32,
    delta_time: f32,
    noise_strength: f32,
    grid_width: u32,
    grid_height: u32,
    kernel_type: u32,        // 0=default, 1=cross, 2=diagonal, 3=spiral
    boundary_mode: u32,      // 0=wrap, 1=clamp, 2=reflect
    map_mode: u32,
    _padding: u32,
}

struct GrayScottApp {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    
    // Ping-pong textures
    texture_a: wgpu::Texture,
    texture_b: wgpu::Texture,
    current_src: bool,
    
    // Compute pipeline
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_a_to_b: wgpu::BindGroup,
    compute_bind_group_b_to_a: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    
    // Paint pipeline
    paint_pipeline: wgpu::ComputePipeline,
    paint_bind_group_a: wgpu::BindGroup,
    paint_bind_group_b: wgpu::BindGroup,
    paint_params_buffer: wgpu::Buffer,
    
    // Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group_a: wgpu::BindGroup,
    render_bind_group_b: wgpu::BindGroup,
    render_params_buffer: wgpu::Buffer,
    gradient_buffer: wgpu::Buffer,
    
    // Parameters
    feed_rate: f32,
    kill_rate: f32,
    diffuse_u: f32,
    diffuse_v: f32,
    delta_time: f32,
    noise_strength: f32,
    kernel_type: u32,
    boundary_mode: u32,
    map_mode: bool,
    paused: bool,
    mouse_pos: Option<(f32, f32)>,
    mouse_down: bool,
    steps_per_frame: u32,
    color_palette: u32,
    
    // View controls
    zoom: f32,
    pan_x: f32,
    pan_y: f32,
    emboss_enabled: bool,
}

impl GrayScottApp {
    async fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        // Create wgpu instance with WebGPU backend (needed for compute shaders)
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        // For WebGPU backend in wgpu 27, the canvas must be the ONLY canvas in the document
        // or we need to use create_surface_unsafe. Let's use unsafe with proper canvas reference.
        let surface = unsafe {
            // WebGPU backend expects the canvas element to be accessible
            // We'll pass it as the raw window handle
            use std::ptr::NonNull;
            
            // Get canvas as JS object
            let js_value = wasm_bindgen::JsValue::from(canvas.clone());
            let canvas_ptr = &js_value as *const wasm_bindgen::JsValue as *mut std::ffi::c_void;
            let nn_ptr = NonNull::new(canvas_ptr).ok_or("Null canvas pointer")?;
            
            let target = wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: raw_window_handle::RawDisplayHandle::Web(
                    raw_window_handle::WebDisplayHandle::new()
                ),
                raw_window_handle: raw_window_handle::RawWindowHandle::WebCanvas(
                    raw_window_handle::WebCanvasWindowHandle::new(nn_ptr)
                ),
            };
            
            instance.create_surface_unsafe(target)
                .map_err(|e| format!("Failed to create surface: {:?}", e))?
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find adapter: {:?}", e))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: Default::default(),
                    experimental_features: Default::default(),
                    trace: Default::default(),
                },
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let width = canvas.width();
        let height = canvas.height();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Create simulation textures
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("Simulation Texture"),
            size: wgpu::Extent3d {
                width: GRID_WIDTH,
                height: GRID_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let texture_a = device.create_texture(&texture_desc);
        let texture_b = device.create_texture(&texture_desc);

        // Initialize texture A with pattern
        let mut init_data = vec![0.0f32; (GRID_WIDTH * GRID_HEIGHT * 2) as usize];
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                let idx = ((y * GRID_WIDTH + x) * 2) as usize;
                init_data[idx] = 1.0; // U starts at 1
                init_data[idx + 1] = 0.0; // V starts at 0
                
                // Add some initial disturbance in the center
                let dx = x as i32 - GRID_WIDTH as i32 / 2;
                let dy = y as i32 - GRID_HEIGHT as i32 / 2;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < 400 {
                    init_data[idx + 1] = 1.0;
                }
            }
        }

        queue.write_texture(
            texture_a.as_image_copy(),
            bytemuck::cast_slice(&init_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(GRID_WIDTH * 8),
                rows_per_image: Some(GRID_HEIGHT),
            },
            wgpu::Extent3d {
                width: GRID_WIDTH,
                height: GRID_HEIGHT,
                depth_or_array_layers: 1,
            },
        );

        // Create parameter buffer
        let params = SimParams {
            feed_rate: 0.055,
            kill_rate: 0.062,
            diffuse_u: 1.0,         // DA - standard value from tutorial
            diffuse_v: 0.5,         // DB - standard value from tutorial
            delta_time: 1.0,
            noise_strength: 0.0,
            grid_width: GRID_WIDTH,
            grid_height: GRID_HEIGHT,
            kernel_type: 0,         // Default kernel
            boundary_mode: 0,       // Wrap (toroidal)
            map_mode: 0,
            _padding: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create compute shader module
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        // Create compute bind group layout (must match compute.wgsl bindings)
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // binding 0: texture_src
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 1: texture_dst
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // binding 2: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 3: params uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create sampler for texture reads - Nearest for Rg32Float (doesn't support filtering)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create compute bind groups
        let texture_a_view = texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_b_view = texture_b.create_view(&wgpu::TextureViewDescriptor::default());

        let compute_bind_group_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A->B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group B->A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create paint shader and pipeline
        let paint_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Paint Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("paint.wgsl").into()),
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PaintParams {
            center_x: f32,
            center_y: f32,
        }

        let paint_params = PaintParams {
            center_x: 0.0,
            center_y: 0.0,
        };

        let paint_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Paint Params Buffer"),
            contents: bytemuck::cast_slice(&[paint_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let paint_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Paint Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let paint_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Paint Pipeline Layout"),
            bind_group_layouts: &[&paint_bind_group_layout],
            push_constant_ranges: &[],
        });

        let paint_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Paint Pipeline"),
            layout: Some(&paint_pipeline_layout),
            module: &paint_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let paint_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Paint Bind Group A"),
            layout: &paint_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: paint_params_buffer.as_entire_binding(),
                },
            ],
        });

        let paint_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Paint Bind Group B"),
            layout: &paint_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: paint_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create render shader module
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
        });

        // Create render bind group layout
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RenderParams {
            color_palette: u32,
            emboss_enabled: u32,
            boundary_mode: u32,
            _padding: u32,
            zoom: f32,
            pan_x: f32,
            pan_y: f32,
        }

        let render_params = RenderParams {
            color_palette: 0,
            emboss_enabled: 1,  // Default to enabled
            boundary_mode: 0,
            _padding: 0,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
        };

        let render_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Render Params Buffer"),
            contents: bytemuck::cast_slice(&[render_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create default gradient (Rainbow)
        let mut gradient_stops = [GradientStop {
            position: 0.0,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
            color: [0.0, 0.0, 0.0, 1.0],
        }; MAX_GRADIENT_STOPS];
        
        gradient_stops[0] = GradientStop { position: 0.0, _padding1: 0.0, _padding2: 0.0, _padding3: 0.0, color: [0.2, 0.0, 0.3, 1.0] }; // Dark purple
        gradient_stops[1] = GradientStop { position: 0.2, _padding1: 0.0, _padding2: 0.0, _padding3: 0.0, color: [0.5, 0.0, 1.0, 1.0] }; // Purple
        gradient_stops[2] = GradientStop { position: 0.4, _padding1: 0.0, _padding2: 0.0, _padding3: 0.0, color: [0.0, 0.5, 1.0, 1.0] }; // Blue
        gradient_stops[3] = GradientStop { position: 0.6, _padding1: 0.0, _padding2: 0.0, _padding3: 0.0, color: [0.0, 1.0, 0.8, 1.0] }; // Cyan
        gradient_stops[4] = GradientStop { position: 0.8, _padding1: 0.0, _padding2: 0.0, _padding3: 0.0, color: [1.0, 0.3, 0.0, 1.0] }; // Orange
        gradient_stops[5] = GradientStop { position: 1.0, _padding1: 0.0, _padding2: 0.0, _padding3: 0.0, color: [1.0, 0.0, 0.0, 1.0] }; // Red
        
        let gradient_data = GradientData {
            stops: gradient_stops,
            num_stops: 6,
            _padding: [0, 0, 0],
            _final_padding: [0.0, 0.0, 0.0, 0.0],
        };
        
        let gradient_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gradient Buffer"),
            contents: bytemuck::cast_slice(&[gradient_data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Create render bind groups
        let render_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group A"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: render_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gradient_buffer.as_entire_binding(),
                },
            ],
        });

        let render_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group B"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: render_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gradient_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            texture_a,
            texture_b,
            current_src: true,
            compute_pipeline,
            compute_bind_group_a_to_b,
            compute_bind_group_b_to_a,
            params_buffer,
            paint_pipeline,
            paint_bind_group_a,
            paint_bind_group_b,
            paint_params_buffer,
            render_pipeline,
            render_bind_group_a,
            render_bind_group_b,
            render_params_buffer,
            gradient_buffer,
            feed_rate: 0.055,
            kill_rate: 0.062,
            diffuse_u: 1.0,         // DA - standard value from tutorial
            diffuse_v: 0.5,         // DB - standard value from tutorial
            delta_time: 1.0,
            noise_strength: 0.0,
            kernel_type: 0,         // Default kernel
            boundary_mode: 2,       // Reflect (Mirror)
            map_mode: false,
            paused: false,
            mouse_pos: None,
            mouse_down: false,
            steps_per_frame: 8,
            color_palette: 0,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            emboss_enabled: true,
        })
    }

    fn update_params(&mut self) {
        let params = SimParams {
            feed_rate: self.feed_rate,
            kill_rate: self.kill_rate,
            diffuse_u: self.diffuse_u,
            diffuse_v: self.diffuse_v,
            delta_time: self.delta_time,
            noise_strength: self.noise_strength,
            grid_width: GRID_WIDTH,
            grid_height: GRID_HEIGHT,
            kernel_type: self.kernel_type,
            boundary_mode: self.boundary_mode,
            map_mode: if self.map_mode { 1 } else { 0 },
            _padding: 0,
        };
        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    fn paint_at(&mut self, x: f32, y: f32) {
        // Apply the SAME transformation the shader uses to convert screen coords to texture coords
        // This is exactly what the shader does in fs_main:
        // 1. zoomed_coords = (screen_coords - 0.5) / zoom + 0.5
        // 2. texture_coords = zoomed_coords + pan
        
        let zoomed_x = (x - 0.5) / self.zoom + 0.5;
        let zoomed_y = (y - 0.5) / self.zoom + 0.5;
        
        let tx = zoomed_x + self.pan_x;
        let ty = zoomed_y + self.pan_y;
        
        let grid_x = (tx * GRID_WIDTH as f32) as u32;
        let grid_y = (ty * GRID_HEIGHT as f32) as u32;
        
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PaintParams {
            center_x: f32,
            center_y: f32,
        }

        let paint_params = PaintParams {
            center_x: grid_x as f32,
            center_y: grid_y as f32,
        };

        self.queue.write_buffer(
            &self.paint_params_buffer,
            0,
            bytemuck::cast_slice(&[paint_params]),
        );

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Paint Encoder"),
        });

        {
            let mut paint_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Paint Pass"),
                timestamp_writes: None,
            });

            paint_pass.set_pipeline(&self.paint_pipeline);
            
            let bind_group = if self.current_src {
                &self.paint_bind_group_a
            } else {
                &self.paint_bind_group_b
            };
            
            paint_pass.set_bind_group(0, bind_group, &[]);
            paint_pass.dispatch_workgroups(GRID_WIDTH / 8, GRID_HEIGHT / 8, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        
        // Toggle since we wrote to the opposite texture
        self.current_src = !self.current_src;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.paused {
            // Run multiple simulation steps per frame
            for _ in 0..self.steps_per_frame {
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });

                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Compute Pass"),
                        timestamp_writes: None,
                    });

                    compute_pass.set_pipeline(&self.compute_pipeline);
                    
                    let bind_group = if self.current_src {
                        &self.compute_bind_group_a_to_b
                    } else {
                        &self.compute_bind_group_b_to_a
                    };
                    
                    compute_pass.set_bind_group(0, bind_group, &[]);
                    compute_pass.dispatch_workgroups(GRID_WIDTH / 8, GRID_HEIGHT / 8, 1);
                }

                self.queue.submit(Some(encoder.finish()));
                self.current_src = !self.current_src;
            }
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            
            let bind_group = if self.current_src {
                &self.render_bind_group_a
            } else {
                &self.render_bind_group_b
            };
            
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }
}

thread_local! {
    static APP: RefCell<Option<Rc<RefCell<GrayScottApp>>>> = RefCell::new(None);
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("Failed to initialize logger");
}

#[wasm_bindgen]
pub async fn init_app(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?
        .dyn_into::<HtmlCanvasElement>()?;

    let app = GrayScottApp::new(canvas).await?;
    APP.with(|a| {
        *a.borrow_mut() = Some(Rc::new(RefCell::new(app)));
    });

    Ok(())
}

#[wasm_bindgen]
pub fn render_frame() -> Result<(), JsValue> {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            app.borrow_mut().render()
                .map_err(|e| format!("Render error: {:?}", e).into())
        } else {
            Err("App not initialized".into())
        }
    })
}

#[wasm_bindgen]
pub fn set_feed_rate(value: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.feed_rate = value;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_kill_rate(value: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.kill_rate = value;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_diffuse_u(value: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.diffuse_u = value;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_diffuse_v(value: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.diffuse_v = value;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_paused(paused: bool) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            app.borrow_mut().paused = paused;
        }
    });
}

#[wasm_bindgen]
pub fn reset() {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let app = app.borrow_mut();
        
        let mut init_data = vec![0.0f32; (GRID_WIDTH * GRID_HEIGHT * 2) as usize];
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                let idx = ((y * GRID_WIDTH + x) * 2) as usize;
                init_data[idx] = 1.0;
                init_data[idx + 1] = 0.0;
                
                let dx = x as i32 - GRID_WIDTH as i32 / 2;
                let dy = y as i32 - GRID_HEIGHT as i32 / 2;
                let dist_sq = dx * dx + dy * dy;
                if dist_sq < 400 {
                    init_data[idx + 1] = 1.0;
                }
            }
        }
        
        let target = if app.current_src { &app.texture_a } else { &app.texture_b };
            app.queue.write_texture(
                target.as_image_copy(),
                bytemuck::cast_slice(&init_data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(GRID_WIDTH * 8),
                    rows_per_image: Some(GRID_HEIGHT),
                },
                wgpu::Extent3d {
                    width: GRID_WIDTH,
                    height: GRID_HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
        }
    });
}

#[wasm_bindgen]
pub fn handle_mouse_down(x: f32, y: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.mouse_down = true;
            app.paint_at(x, y);
        }
    });
}

#[wasm_bindgen]
pub fn handle_mouse_up() {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            app.borrow_mut().mouse_down = false;
        }
    });
}

#[wasm_bindgen]
pub fn handle_mouse_move(x: f32, y: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.mouse_pos = Some((x, y));
            if app.mouse_down {
                app.paint_at(x, y);
            }
        }
    });
}

#[wasm_bindgen]
pub fn apply_preset(feed: f32, kill: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.feed_rate = feed;
            app.kill_rate = kill;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_color_palette(palette: u32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            app_mut.color_palette = palette;
            
            update_render_params(&mut app_mut);
        }
    });
}

#[wasm_bindgen]
pub fn set_kernel(kernel: u32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            app_mut.kernel_type = kernel;
            app_mut.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_boundary(boundary: u32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            app_mut.boundary_mode = boundary;
            app_mut.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_noise(strength: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            app_mut.noise_strength = strength;
            app_mut.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_gradient(positions: &[f32], colors: &[f32]) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let app_mut = app.borrow_mut();
            
            // Validate input: positions and colors must be aligned
            let num_stops = positions.len().min(colors.len() / 4).min(MAX_GRADIENT_STOPS);
            
            let mut gradient_stops = [GradientStop {
                position: 0.0,
                _padding1: 0.0,
                _padding2: 0.0,
                _padding3: 0.0,
                color: [0.0, 0.0, 0.0, 1.0],
            }; MAX_GRADIENT_STOPS];
            
            for i in 0..num_stops {
                gradient_stops[i] = GradientStop {
                    position: positions[i],
                    _padding1: 0.0,
                    _padding2: 0.0,
                    _padding3: 0.0,
                    color: [
                        colors[i * 4],
                        colors[i * 4 + 1],
                        colors[i * 4 + 2],
                        colors[i * 4 + 3],
                    ],
                };
            }
            
            let gradient_data = GradientData {
                stops: gradient_stops,
                num_stops: num_stops as u32,
                _padding: [0, 0, 0],
                _final_padding: [0.0, 0.0, 0.0, 0.0],
            };
            
            app_mut.queue.write_buffer(&app_mut.gradient_buffer, 0, bytemuck::cast_slice(&[gradient_data]));
        }
    });
}

#[wasm_bindgen]
pub fn set_map_mode(enabled: bool) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            app_mut.map_mode = enabled;
            app_mut.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_emboss(enabled: bool) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            app_mut.emboss_enabled = enabled;
            update_render_params(&mut app_mut);
        }
    });
}

fn update_render_params(app: &mut GrayScottApp) {
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct RenderParams {
        color_palette: u32,
        emboss_enabled: u32,
        boundary_mode: u32,
        _padding: u32,
        zoom: f32,
        pan_x: f32,
        pan_y: f32,
    }
    
    let render_params = RenderParams {
        color_palette: app.color_palette,
        emboss_enabled: if app.emboss_enabled { 1 } else { 0 },
        boundary_mode: app.boundary_mode,
        _padding: 0,
        zoom: app.zoom,
        pan_x: app.pan_x,
        pan_y: app.pan_y,
    };
    
    app.queue.write_buffer(
        &app.render_params_buffer,
        0,
        bytemuck::cast_slice(&[render_params]),
    );
}

#[wasm_bindgen]
pub fn set_zoom(zoom: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            app_mut.zoom = zoom.max(1.0); // Minimum 1.0 (can't zoom out past full view), no maximum
            update_render_params(&mut app_mut);
        }
    });
}

#[wasm_bindgen]
pub fn set_pan(x: f32, y: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app_mut = app.borrow_mut();
            // Clamp pan to reasonable range
            app_mut.pan_x = x.max(-1.0).min(1.0);
            app_mut.pan_y = y.max(-1.0).min(1.0);
            update_render_params(&mut app_mut);
        }
    });
}

#[wasm_bindgen]
pub fn set_steps_per_frame(steps: u32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            app.borrow_mut().steps_per_frame = steps;
        }
    });
}

#[wasm_bindgen]
pub fn set_delta_time(value: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.delta_time = value;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_diffusion_u(value: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.diffuse_u = value;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn set_diffusion_v(value: f32) {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            app.diffuse_v = value;
            app.update_params();
        }
    });
}

#[wasm_bindgen]
pub fn step_once() {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            let was_paused = app.paused;
            app.paused = false;
            let _ = app.render();
            app.paused = was_paused;
        }
    });
}

#[wasm_bindgen]
pub fn clear_canvas() {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            let mut init_data = vec![0.0f32; (GRID_WIDTH * GRID_HEIGHT * 2) as usize];
            for i in (0..init_data.len()).step_by(2) {
                init_data[i] = 1.0;  // U = 1
                init_data[i + 1] = 0.0;  // V = 0
            }
            
            app.queue.write_texture(
                app.texture_a.as_image_copy(),
                bytemuck::cast_slice(&init_data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(GRID_WIDTH * 8),
                    rows_per_image: Some(GRID_HEIGHT),
                },
                wgpu::Extent3d {
                    width: GRID_WIDTH,
                    height: GRID_HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
            app.current_src = true;
        }
    });
}

#[wasm_bindgen]
pub fn add_random_blobs() {
    APP.with(|a| {
        if let Some(app) = a.borrow().as_ref() {
            let mut app = app.borrow_mut();
            let mut init_data = vec![0.0f32; (GRID_WIDTH * GRID_HEIGHT * 2) as usize];
            
            // Fill with base state
            for i in (0..init_data.len()).step_by(2) {
                init_data[i] = 1.0;  // U = 1
                init_data[i + 1] = 0.0;  // V = 0
            }
            
            // Add random blobs
            for _ in 0..15 {
                let cx = (js_sys::Math::random() * GRID_WIDTH as f64) as u32;
                let cy = (js_sys::Math::random() * GRID_HEIGHT as f64) as u32;
                let radius = (js_sys::Math::random() * 30.0 + 10.0) as i32;
                
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let px = cx as i32 + dx;
                        let py = cy as i32 + dy;
                        
                        if px >= 0 && px < GRID_WIDTH as i32 && py >= 0 && py < GRID_HEIGHT as i32 {
                            let dist_sq = dx * dx + dy * dy;
                            if dist_sq <= radius * radius {
                                let idx = ((py as u32 * GRID_WIDTH + px as u32) * 2) as usize;
                                init_data[idx + 1] = 1.0;  // V = 1 in blob
                            }
                        }
                    }
                }
            }
            
            app.queue.write_texture(
                app.texture_a.as_image_copy(),
                bytemuck::cast_slice(&init_data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(GRID_WIDTH * 8),
                    rows_per_image: Some(GRID_HEIGHT),
                },
                wgpu::Extent3d {
                    width: GRID_WIDTH,
                    height: GRID_HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
            app.current_src = true;
        }
    });
}
