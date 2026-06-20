//! vk_probe — minimal Vulkan compute SSBO write-back probe.
//!
//! Purpose: confirm the egg/Venus blob-sync *direction* bug independently of
//! llama.cpp. A compute shader writes `data[i] = i + 1` into a storage buffer;
//! the host maps the same memory and reads it back. If the GPU's writes never
//! reach the host's view, we read zeros.
//!
//! The host's hypothesis (FOR_VM.md): egg keeps a second copy of each
//! host-visible blob and picks the memcpy direction by SIZE — blobs >500 KB are
//! assumed GPU-written (metal→shm), smaller ones guest-written (shm→metal). So
//! correctness should FLIP as the buffer crosses ~500 KB. This probe runs the
//! same shader at several sizes straddling that threshold and reports each.
//!
//! No Vulkan dev headers/loader-dev are installed, so we dlopen libvulkan.so.1
//! and bootstrap via vkGetInstanceProcAddr. SPIR-V is compiled at build time by
//! tools/vk-probe/build (glslangValidator) and embedded here.
//!
//! Usage:
//!   vk_probe [sizes_in_KiB ...]         # default: 64 256 512 1024 4096
//!   VK_PROBE_DEVICE=venus|llvmpipe      # substring to select the device
//!
//! Run it on `venus` (the bug) and `llvmpipe` (the control) and compare.

use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;

const SPIRV: &[u8] = include_bytes!("../../tools/vk-probe/probe.spv");
// Read-modify-write shader (data[i] += 1) — tests the host→GPU input direction.
const SPIRV_RMW: &[u8] = include_bytes!("../../tools/vk-probe/probe_rmw.spv");

// ---- handles (dispatchable = pointer-sized; non-dispatchable = u64) ----
type Instance = usize;
type PhysicalDevice = usize;
type Device = usize;
type Queue = usize;
type CmdBuffer = usize;

// ---- VkStructureType values we use ----
const ST_APP_INFO: i32 = 0;
const ST_INSTANCE_CI: i32 = 1;
const ST_DEVICE_QUEUE_CI: i32 = 2;
const ST_DEVICE_CI: i32 = 3;
const ST_SUBMIT: i32 = 4;
const ST_MEMORY_ALLOC: i32 = 5;
const ST_FENCE_CI: i32 = 8;
const ST_BUFFER_CI: i32 = 12;
const ST_SHADER_MODULE_CI: i32 = 16;
const ST_PIPELINE_SHADER_STAGE_CI: i32 = 18;
const ST_COMPUTE_PIPELINE_CI: i32 = 29;
const ST_PIPELINE_LAYOUT_CI: i32 = 30;
const ST_DSL_CI: i32 = 32;
const ST_DESC_POOL_CI: i32 = 33;
const ST_DESC_SET_ALLOC: i32 = 34;
const ST_WRITE_DESC_SET: i32 = 35;
const ST_CMD_POOL_CI: i32 = 39;
const ST_CMD_BUFFER_ALLOC: i32 = 40;
const ST_CMD_BUFFER_BEGIN: i32 = 42;
const ST_BUFFER_MEMORY_BARRIER: i32 = 44;

const QUEUE_COMPUTE_BIT: u32 = 0x2;
const USAGE_STORAGE_BUFFER: u32 = 0x20;
const MEM_HOST_VISIBLE: u32 = 0x2;
const MEM_HOST_COHERENT: u32 = 0x4;
const DESC_STORAGE_BUFFER: u32 = 7;
const SHADER_STAGE_COMPUTE: u32 = 0x20;
const BIND_POINT_COMPUTE: u32 = 1;
const CB_ONE_TIME_SUBMIT: u32 = 0x1;
const ACCESS_SHADER_WRITE: u32 = 0x40;
const ACCESS_HOST_READ: u32 = 0x2000;
const STAGE_COMPUTE_SHADER: u32 = 0x800;
const STAGE_HOST: u32 = 0x4000;
const QFI_IGNORED: u32 = u32::MAX;
const WHOLE_SIZE: u64 = u64::MAX;

#[repr(C)]
struct AppInfo {
    s_type: i32,
    p_next: *const c_void,
    app_name: *const c_char,
    app_ver: u32,
    engine_name: *const c_char,
    engine_ver: u32,
    api_ver: u32,
}
#[repr(C)]
struct InstanceCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    app_info: *const AppInfo,
    layer_count: u32,
    layer_names: *const *const c_char,
    ext_count: u32,
    ext_names: *const *const c_char,
}
#[repr(C)]
struct QueueCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
    queue_count: u32,
    priorities: *const f32,
}
#[repr(C)]
struct DeviceCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    queue_ci_count: u32,
    queue_cis: *const QueueCI,
    layer_count: u32,
    layer_names: *const *const c_char,
    ext_count: u32,
    ext_names: *const *const c_char,
    features: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryType {
    property_flags: u32,
    heap_index: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryHeap {
    size: u64,
    flags: u32,
}
#[repr(C)]
struct MemoryProperties {
    type_count: u32,
    types: [MemoryType; 32],
    heap_count: u32,
    heaps: [MemoryHeap; 16],
}
#[repr(C)]
struct MemoryRequirements {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
}
#[repr(C)]
struct QueueFamilyProperties {
    queue_flags: u32,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: [u32; 3],
}
#[repr(C)]
struct BufferCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharing_mode: u32,
    qfi_count: u32,
    qfi: *const u32,
}
#[repr(C)]
struct MemoryAllocateInfo {
    s_type: i32,
    p_next: *const c_void,
    allocation_size: u64,
    memory_type_index: u32,
}
#[repr(C)]
struct DSLBinding {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
    immutable: *const c_void,
}
#[repr(C)]
struct DSLCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    binding_count: u32,
    bindings: *const DSLBinding,
}
#[repr(C)]
struct DescPoolSize {
    type_: u32,
    count: u32,
}
#[repr(C)]
struct DescPoolCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    max_sets: u32,
    pool_size_count: u32,
    pool_sizes: *const DescPoolSize,
}
#[repr(C)]
struct DescSetAllocInfo {
    s_type: i32,
    p_next: *const c_void,
    pool: u64,
    set_count: u32,
    layouts: *const u64,
}
#[repr(C)]
struct DescBufferInfo {
    buffer: u64,
    offset: u64,
    range: u64,
}
#[repr(C)]
struct WriteDescSet {
    s_type: i32,
    p_next: *const c_void,
    dst_set: u64,
    dst_binding: u32,
    dst_array: u32,
    descriptor_count: u32,
    descriptor_type: u32,
    image_info: *const c_void,
    buffer_info: *const DescBufferInfo,
    texel: *const c_void,
}
#[repr(C)]
struct PipelineLayoutCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    set_layout_count: u32,
    set_layouts: *const u64,
    push_count: u32,
    push_ranges: *const c_void,
}
#[repr(C)]
struct ShaderModuleCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    code_size: usize,
    code: *const u32,
}
#[repr(C)]
struct ShaderStageCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    stage: u32,
    module: u64,
    name: *const c_char,
    spec: *const c_void,
}
#[repr(C)]
struct ComputePipelineCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    stage: ShaderStageCI,
    layout: u64,
    base_handle: u64,
    base_index: i32,
}
#[repr(C)]
struct CmdPoolCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    queue_family_index: u32,
}
#[repr(C)]
struct CmdBufferAllocInfo {
    s_type: i32,
    p_next: *const c_void,
    pool: u64,
    level: u32,
    count: u32,
}
#[repr(C)]
struct CmdBufferBeginInfo {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
    inheritance: *const c_void,
}
#[repr(C)]
struct BufferMemoryBarrier {
    s_type: i32,
    p_next: *const c_void,
    src_access: u32,
    dst_access: u32,
    src_qfi: u32,
    dst_qfi: u32,
    buffer: u64,
    offset: u64,
    size: u64,
}
#[repr(C)]
struct SubmitInfo {
    s_type: i32,
    p_next: *const c_void,
    wait_count: u32,
    wait_sems: *const u64,
    wait_stages: *const u32,
    cb_count: u32,
    cbs: *const usize,
    signal_count: u32,
    signal_sems: *const u64,
}
#[repr(C)]
struct FenceCI {
    s_type: i32,
    p_next: *const c_void,
    flags: u32,
}

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

type GIPA = extern "C" fn(Instance, *const c_char) -> *const c_void;

unsafe fn pfn(gipa: GIPA, inst: Instance, name: &str) -> *const c_void {
    let c = CString::new(name).unwrap();
    let p = gipa(inst, c.as_ptr());
    if p.is_null() {
        panic!("vkGetInstanceProcAddr returned null for {name}");
    }
    p
}

macro_rules! load {
    ($gipa:expr, $inst:expr, $name:expr, $t:ty) => {
        core::mem::transmute::<*const c_void, $t>(pfn($gipa, $inst, $name))
    };
}

fn check(r: i32, what: &str) {
    if r != 0 {
        eprintln!("error: {what} -> VkResult {r}");
        std::process::exit(1);
    }
}

fn main() {
    let sizes_kib: Vec<u64> = {
        let a: Vec<u64> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
        if a.is_empty() {
            vec![64, 256, 512, 1024, 4096]
        } else {
            a
        }
    };
    let want = std::env::var("VK_PROBE_DEVICE").unwrap_or_default().to_lowercase();

    unsafe {
        let lib = dlopen(CString::new("libvulkan.so.1").unwrap().as_ptr(), RTLD_NOW);
        if lib.is_null() {
            eprintln!("error: cannot dlopen libvulkan.so.1");
            std::process::exit(1);
        }
        let gipa: GIPA = core::mem::transmute(dlsym(
            lib,
            CString::new("vkGetInstanceProcAddr").unwrap().as_ptr(),
        ));

        // ---- instance ----
        let create_instance: extern "C" fn(*const InstanceCI, *const c_void, *mut Instance) -> i32 =
            load!(gipa, 0, "vkCreateInstance", _);
        let app = AppInfo {
            s_type: ST_APP_INFO,
            p_next: ptr::null(),
            app_name: c"vk_probe".as_ptr(),
            app_ver: 0,
            engine_name: c"none".as_ptr(),
            engine_ver: 0,
            api_ver: (1 << 22) | (1 << 12), // 1.1
        };
        let ici = InstanceCI {
            s_type: ST_INSTANCE_CI,
            p_next: ptr::null(),
            flags: 0,
            app_info: &app,
            layer_count: 0,
            layer_names: ptr::null(),
            ext_count: 0,
            ext_names: ptr::null(),
        };
        let mut inst: Instance = 0;
        check(create_instance(&ici, ptr::null(), &mut inst), "vkCreateInstance");

        let enum_devices: extern "C" fn(Instance, *mut u32, *mut PhysicalDevice) -> i32 =
            load!(gipa, inst, "vkEnumeratePhysicalDevices", _);
        let get_props: extern "C" fn(PhysicalDevice, *mut u8) =
            load!(gipa, inst, "vkGetPhysicalDeviceProperties", _);
        let get_qf: extern "C" fn(PhysicalDevice, *mut u32, *mut QueueFamilyProperties) =
            load!(gipa, inst, "vkGetPhysicalDeviceQueueFamilyProperties", _);
        let get_mem: extern "C" fn(PhysicalDevice, *mut MemoryProperties) =
            load!(gipa, inst, "vkGetPhysicalDeviceMemoryProperties", _);

        let mut n_dev = 0u32;
        check(enum_devices(inst, &mut n_dev, ptr::null_mut()), "enumerate count");
        let mut devs = vec![0usize; n_dev as usize];
        check(enum_devices(inst, &mut n_dev, devs.as_mut_ptr()), "enumerate list");

        // device props buffer: VkPhysicalDeviceProperties; deviceName[256] at offset 20.
        let dev_name = |pd: PhysicalDevice| -> String {
            let mut buf = vec![0u8; 1024];
            get_props(pd, buf.as_mut_ptr());
            let name = &buf[20..20 + 256];
            let end = name.iter().position(|&b| b == 0).unwrap_or(256);
            String::from_utf8_lossy(&name[..end]).into_owned()
        };

        println!("Vulkan devices:");
        for (i, &pd) in devs.iter().enumerate() {
            println!("  [{i}] {}", dev_name(pd));
        }
        let chosen = devs
            .iter()
            .copied()
            .find(|&pd| want.is_empty() || dev_name(pd).to_lowercase().contains(&want))
            .unwrap_or(devs[0]);
        println!("selected: {}\n", dev_name(chosen));

        // ---- compute queue family ----
        let mut n_qf = 0u32;
        get_qf(chosen, &mut n_qf, ptr::null_mut());
        let mut qfs: Vec<QueueFamilyProperties> = (0..n_qf)
            .map(|_| QueueFamilyProperties {
                queue_flags: 0,
                queue_count: 0,
                timestamp_valid_bits: 0,
                min_image_transfer_granularity: [0; 3],
            })
            .collect();
        get_qf(chosen, &mut n_qf, qfs.as_mut_ptr());
        let qfi = qfs
            .iter()
            .position(|q| q.queue_flags & QUEUE_COMPUTE_BIT != 0)
            .expect("no compute queue family") as u32;

        // ---- memory types ----
        let mut mp = MemoryProperties {
            type_count: 0,
            types: [MemoryType { property_flags: 0, heap_index: 0 }; 32],
            heap_count: 0,
            heaps: [MemoryHeap { size: 0, flags: 0 }; 16],
        };
        get_mem(chosen, &mut mp);

        // ---- device + queue ----
        let create_device: extern "C" fn(PhysicalDevice, *const DeviceCI, *const c_void, *mut Device) -> i32 =
            load!(gipa, inst, "vkCreateDevice", _);
        let prio = [1.0f32];
        let qci = QueueCI {
            s_type: ST_DEVICE_QUEUE_CI,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index: qfi,
            queue_count: 1,
            priorities: prio.as_ptr(),
        };
        let dci = DeviceCI {
            s_type: ST_DEVICE_CI,
            p_next: ptr::null(),
            flags: 0,
            queue_ci_count: 1,
            queue_cis: &qci,
            layer_count: 0,
            layer_names: ptr::null(),
            ext_count: 0,
            ext_names: ptr::null(),
            features: ptr::null(),
        };
        let mut dev: Device = 0;
        check(create_device(chosen, &dci, ptr::null(), &mut dev), "vkCreateDevice");

        let get_queue: extern "C" fn(Device, u32, u32, *mut Queue) =
            load!(gipa, inst, "vkGetDeviceQueue", _);
        let mut queue: Queue = 0;
        get_queue(dev, qfi, 0, &mut queue);

        // ---- device functions ----
        let create_buffer: extern "C" fn(Device, *const BufferCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreateBuffer", _);
        let buf_mem_req: extern "C" fn(Device, u64, *mut MemoryRequirements) =
            load!(gipa, inst, "vkGetBufferMemoryRequirements", _);
        let alloc_mem: extern "C" fn(Device, *const MemoryAllocateInfo, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkAllocateMemory", _);
        let bind_buf: extern "C" fn(Device, u64, u64, u64) -> i32 =
            load!(gipa, inst, "vkBindBufferMemory", _);
        let map_mem: extern "C" fn(Device, u64, u64, u64, u32, *mut *mut c_void) -> i32 =
            load!(gipa, inst, "vkMapMemory", _);
        let create_dsl: extern "C" fn(Device, *const DSLCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreateDescriptorSetLayout", _);
        let create_pool: extern "C" fn(Device, *const DescPoolCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreateDescriptorPool", _);
        let alloc_sets: extern "C" fn(Device, *const DescSetAllocInfo, *mut u64) -> i32 =
            load!(gipa, inst, "vkAllocateDescriptorSets", _);
        let update_sets: extern "C" fn(Device, u32, *const WriteDescSet, u32, *const c_void) =
            load!(gipa, inst, "vkUpdateDescriptorSets", _);
        let create_pl_layout: extern "C" fn(Device, *const PipelineLayoutCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreatePipelineLayout", _);
        let create_shader: extern "C" fn(Device, *const ShaderModuleCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreateShaderModule", _);
        let create_compute_pipelines: extern "C" fn(Device, u64, u32, *const ComputePipelineCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreateComputePipelines", _);
        let create_cmd_pool: extern "C" fn(Device, *const CmdPoolCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreateCommandPool", _);
        let alloc_cmd: extern "C" fn(Device, *const CmdBufferAllocInfo, *mut CmdBuffer) -> i32 =
            load!(gipa, inst, "vkAllocateCommandBuffers", _);
        let begin_cmd: extern "C" fn(CmdBuffer, *const CmdBufferBeginInfo) -> i32 =
            load!(gipa, inst, "vkBeginCommandBuffer", _);
        let cmd_bind_pipeline: extern "C" fn(CmdBuffer, u32, u64) =
            load!(gipa, inst, "vkCmdBindPipeline", _);
        let cmd_bind_sets: extern "C" fn(CmdBuffer, u32, u64, u32, u32, *const u64, u32, *const u32) =
            load!(gipa, inst, "vkCmdBindDescriptorSets", _);
        let cmd_dispatch: extern "C" fn(CmdBuffer, u32, u32, u32) =
            load!(gipa, inst, "vkCmdDispatch", _);
        let cmd_barrier: extern "C" fn(CmdBuffer, u32, u32, u32, u32, *const c_void, u32, *const BufferMemoryBarrier, u32, *const c_void) =
            load!(gipa, inst, "vkCmdPipelineBarrier", _);
        let end_cmd: extern "C" fn(CmdBuffer) -> i32 = load!(gipa, inst, "vkEndCommandBuffer", _);
        let create_fence: extern "C" fn(Device, *const FenceCI, *const c_void, *mut u64) -> i32 =
            load!(gipa, inst, "vkCreateFence", _);
        let queue_submit: extern "C" fn(Queue, u32, *const SubmitInfo, u64) -> i32 =
            load!(gipa, inst, "vkQueueSubmit", _);
        let wait_fences: extern "C" fn(Device, u32, *const u64, u32, u64) -> i32 =
            load!(gipa, inst, "vkWaitForFences", _);

        // shared shader module + descriptor layout + pipeline (reused per size)
        let smci = ShaderModuleCI {
            s_type: ST_SHADER_MODULE_CI,
            p_next: ptr::null(),
            flags: 0,
            code_size: SPIRV.len(),
            code: SPIRV.as_ptr() as *const u32,
        };
        let mut shader = 0u64;
        check(create_shader(dev, &smci, ptr::null(), &mut shader), "vkCreateShaderModule");

        let binding = DSLBinding {
            binding: 0,
            descriptor_type: DESC_STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: SHADER_STAGE_COMPUTE,
            immutable: ptr::null(),
        };
        let dslci = DSLCI {
            s_type: ST_DSL_CI,
            p_next: ptr::null(),
            flags: 0,
            binding_count: 1,
            bindings: &binding,
        };
        let mut dsl = 0u64;
        check(create_dsl(dev, &dslci, ptr::null(), &mut dsl), "vkCreateDescriptorSetLayout");

        let plci = PipelineLayoutCI {
            s_type: ST_PIPELINE_LAYOUT_CI,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            set_layouts: &dsl,
            push_count: 0,
            push_ranges: ptr::null(),
        };
        let mut pl_layout = 0u64;
        check(create_pl_layout(dev, &plci, ptr::null(), &mut pl_layout), "vkCreatePipelineLayout");

        let cpci = ComputePipelineCI {
            s_type: ST_COMPUTE_PIPELINE_CI,
            p_next: ptr::null(),
            flags: 0,
            stage: ShaderStageCI {
                s_type: ST_PIPELINE_SHADER_STAGE_CI,
                p_next: ptr::null(),
                flags: 0,
                stage: SHADER_STAGE_COMPUTE,
                module: shader,
                name: c"main".as_ptr(),
                spec: ptr::null(),
            },
            layout: pl_layout,
            base_handle: 0,
            base_index: -1,
        };
        let mut pipeline = 0u64;
        check(
            create_compute_pipelines(dev, 0, 1, &cpci, ptr::null(), &mut pipeline),
            "vkCreateComputePipelines",
        );

        // second pipeline: read-modify-write (data[i] += 1), shares pl_layout
        let smci_rmw = ShaderModuleCI {
            s_type: ST_SHADER_MODULE_CI,
            p_next: ptr::null(),
            flags: 0,
            code_size: SPIRV_RMW.len(),
            code: SPIRV_RMW.as_ptr() as *const u32,
        };
        let mut shader_rmw = 0u64;
        check(create_shader(dev, &smci_rmw, ptr::null(), &mut shader_rmw), "vkCreateShaderModule(rmw)");
        let cpci_rmw = ComputePipelineCI {
            s_type: ST_COMPUTE_PIPELINE_CI,
            p_next: ptr::null(),
            flags: 0,
            stage: ShaderStageCI {
                s_type: ST_PIPELINE_SHADER_STAGE_CI,
                p_next: ptr::null(),
                flags: 0,
                stage: SHADER_STAGE_COMPUTE,
                module: shader_rmw,
                name: c"main".as_ptr(),
                spec: ptr::null(),
            },
            layout: pl_layout,
            base_handle: 0,
            base_index: -1,
        };
        let mut pipeline_rmw = 0u64;
        check(
            create_compute_pipelines(dev, 0, 1, &cpci_rmw, ptr::null(), &mut pipeline_rmw),
            "vkCreateComputePipelines(rmw)",
        );

        let cmd_pool_ci = CmdPoolCI {
            s_type: ST_CMD_POOL_CI,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index: qfi,
        };
        let mut cmd_pool = 0u64;
        check(create_cmd_pool(dev, &cmd_pool_ci, ptr::null(), &mut cmd_pool), "vkCreateCommandPool");

        println!("{:>9} {:>10}  memtype  result", "size", "bytes");
        let mut any_fail = false;
        for kib in &sizes_kib {
            let n_elems = (kib * 1024 / 4) as u32; // u32 elements
            let n_elems = (n_elems / 64) * 64; // multiple of local_size_x
            let bytes = (n_elems as u64) * 4;

            // buffer
            let bci = BufferCI {
                s_type: ST_BUFFER_CI,
                p_next: ptr::null(),
                flags: 0,
                size: bytes,
                usage: USAGE_STORAGE_BUFFER,
                sharing_mode: 0,
                qfi_count: 0,
                qfi: ptr::null(),
            };
            let mut buffer = 0u64;
            check(create_buffer(dev, &bci, ptr::null(), &mut buffer), "vkCreateBuffer");
            let mut req = MemoryRequirements { size: 0, alignment: 0, memory_type_bits: 0 };
            buf_mem_req(dev, buffer, &mut req);

            // pick a HOST_VISIBLE|HOST_COHERENT memory type allowed by the buffer
            let mut mt_index = u32::MAX;
            for i in 0..mp.type_count {
                let t = mp.types[i as usize];
                let ok_bits = req.memory_type_bits & (1 << i) != 0;
                let host = t.property_flags & (MEM_HOST_VISIBLE | MEM_HOST_COHERENT)
                    == (MEM_HOST_VISIBLE | MEM_HOST_COHERENT);
                if ok_bits && host {
                    mt_index = i;
                    break;
                }
            }
            if mt_index == u32::MAX {
                eprintln!("error: no HOST_VISIBLE|HOST_COHERENT memory type");
                std::process::exit(1);
            }
            let mt_flags = mp.types[mt_index as usize].property_flags;

            let mai = MemoryAllocateInfo {
                s_type: ST_MEMORY_ALLOC,
                p_next: ptr::null(),
                allocation_size: req.size,
                memory_type_index: mt_index,
            };
            let mut mem = 0u64;
            check(alloc_mem(dev, &mai, ptr::null(), &mut mem), "vkAllocateMemory");
            check(bind_buf(dev, buffer, mem, 0), "vkBindBufferMemory");

            // map + pre-zero so a missing GPU write reads as 0
            let mut mapped: *mut c_void = ptr::null_mut();
            check(map_mem(dev, mem, 0, WHOLE_SIZE, 0, &mut mapped), "vkMapMemory");
            let slice = std::slice::from_raw_parts_mut(mapped as *mut u32, n_elems as usize);
            for v in slice.iter_mut() {
                *v = 0;
            }

            // descriptor pool + set
            let pool_size = DescPoolSize { type_: DESC_STORAGE_BUFFER, count: 1 };
            let pool_ci = DescPoolCI {
                s_type: ST_DESC_POOL_CI,
                p_next: ptr::null(),
                flags: 0,
                max_sets: 1,
                pool_size_count: 1,
                pool_sizes: &pool_size,
            };
            let mut pool = 0u64;
            check(create_pool(dev, &pool_ci, ptr::null(), &mut pool), "vkCreateDescriptorPool");
            let set_ai = DescSetAllocInfo {
                s_type: ST_DESC_SET_ALLOC,
                p_next: ptr::null(),
                pool,
                set_count: 1,
                layouts: &dsl,
            };
            let mut set = 0u64;
            check(alloc_sets(dev, &set_ai, &mut set), "vkAllocateDescriptorSets");
            let bi = DescBufferInfo { buffer, offset: 0, range: WHOLE_SIZE };
            let write = WriteDescSet {
                s_type: ST_WRITE_DESC_SET,
                p_next: ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array: 0,
                descriptor_count: 1,
                descriptor_type: DESC_STORAGE_BUFFER,
                image_info: ptr::null(),
                buffer_info: &bi,
                texel: ptr::null(),
            };
            update_sets(dev, 1, &write, 0, ptr::null());

            // command buffer
            let cb_ai = CmdBufferAllocInfo {
                s_type: ST_CMD_BUFFER_ALLOC,
                p_next: ptr::null(),
                pool: cmd_pool,
                level: 0,
                count: 1,
            };
            let mut cb: CmdBuffer = 0;
            check(alloc_cmd(dev, &cb_ai, &mut cb), "vkAllocateCommandBuffers");
            let begin = CmdBufferBeginInfo {
                s_type: ST_CMD_BUFFER_BEGIN,
                p_next: ptr::null(),
                flags: CB_ONE_TIME_SUBMIT,
                inheritance: ptr::null(),
            };
            check(begin_cmd(cb, &begin), "vkBeginCommandBuffer");
            cmd_bind_pipeline(cb, BIND_POINT_COMPUTE, pipeline);
            cmd_bind_sets(cb, BIND_POINT_COMPUTE, pl_layout, 0, 1, &set, 0, ptr::null());
            cmd_dispatch(cb, n_elems / 64, 1, 1);
            let barrier = BufferMemoryBarrier {
                s_type: ST_BUFFER_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access: ACCESS_SHADER_WRITE,
                dst_access: ACCESS_HOST_READ,
                src_qfi: QFI_IGNORED,
                dst_qfi: QFI_IGNORED,
                buffer,
                offset: 0,
                size: WHOLE_SIZE,
            };
            cmd_barrier(cb, STAGE_COMPUTE_SHADER, STAGE_HOST, 0, 0, ptr::null(), 1, &barrier, 0, ptr::null());
            check(end_cmd(cb), "vkEndCommandBuffer");

            // submit + wait
            let fci = FenceCI { s_type: ST_FENCE_CI, p_next: ptr::null(), flags: 0 };
            let mut fence = 0u64;
            check(create_fence(dev, &fci, ptr::null(), &mut fence), "vkCreateFence");
            let submit = SubmitInfo {
                s_type: ST_SUBMIT,
                p_next: ptr::null(),
                wait_count: 0,
                wait_sems: ptr::null(),
                wait_stages: ptr::null(),
                cb_count: 1,
                cbs: &cb,
                signal_count: 0,
                signal_sems: ptr::null(),
            };
            check(queue_submit(queue, 1, &submit, fence), "vkQueueSubmit");
            // On Venus the fence can hang (VK_TIMEOUT) — the host's
            // `vkr FATAL CS cmd=39` path. Don't abort: note it and still read
            // back, since the host's 1ms blob-sync timer may have delivered data.
            let wr = wait_fences(dev, 1, &fence, 1, 20_000_000_000);
            let fence_note = if wr == 0 { "" } else { " [fence-timeout]" };

            // --- WRITE test (GPU→host): expected slice[i] == i+1 ---
            let sample = [0usize, 1, 2, 7, (n_elems as usize) / 2, n_elems as usize - 1];
            let w_ok = sample.iter().all(|&i| slice[i] == (i as u32) + 1);

            // --- READ test (host→GPU): pre-fill an input pattern the host writes,
            // run data[i] += 1 on the GPU, expect input+1. If the GPU never sees
            // the host's bytes (input not synced to GPU), it reads 0 → writes 1
            // everywhere → mismatch. This is the weights-upload direction llama needs.
            const TAG: u32 = 0x1000;
            for (i, v) in slice.iter_mut().enumerate() {
                *v = i as u32 + TAG;
            }
            let cb2_ai = CmdBufferAllocInfo {
                s_type: ST_CMD_BUFFER_ALLOC,
                p_next: ptr::null(),
                pool: cmd_pool,
                level: 0,
                count: 1,
            };
            let mut cb2: CmdBuffer = 0;
            check(alloc_cmd(dev, &cb2_ai, &mut cb2), "vkAllocateCommandBuffers(rmw)");
            check(begin_cmd(cb2, &begin), "vkBeginCommandBuffer(rmw)");
            cmd_bind_pipeline(cb2, BIND_POINT_COMPUTE, pipeline_rmw);
            cmd_bind_sets(cb2, BIND_POINT_COMPUTE, pl_layout, 0, 1, &set, 0, ptr::null());
            cmd_dispatch(cb2, n_elems / 64, 1, 1);
            cmd_barrier(cb2, STAGE_COMPUTE_SHADER, STAGE_HOST, 0, 0, ptr::null(), 1, &barrier, 0, ptr::null());
            check(end_cmd(cb2), "vkEndCommandBuffer(rmw)");
            let mut fence2 = 0u64;
            check(create_fence(dev, &fci, ptr::null(), &mut fence2), "vkCreateFence(rmw)");
            let submit2 = SubmitInfo {
                s_type: ST_SUBMIT,
                p_next: ptr::null(),
                wait_count: 0,
                wait_sems: ptr::null(),
                wait_stages: ptr::null(),
                cb_count: 1,
                cbs: &cb2,
                signal_count: 0,
                signal_sems: ptr::null(),
            };
            check(queue_submit(queue, 1, &submit2, fence2), "vkQueueSubmit(rmw)");
            let _ = wait_fences(dev, 1, &fence2, 1, 20_000_000_000);
            let r_ok = sample.iter().all(|&i| slice[i] == (i as u32) + TAG + 1);
            let r_first: Vec<u32> = slice.iter().take(4).copied().collect();

            if !w_ok || !r_ok {
                any_fail = true;
            }
            let v = |ok: bool| if ok { "PASS" } else { "FAIL" };
            println!(
                "{:>7}KiB {:>10}  t{} 0x{:04x}  write={} read={}  r_first={:?}{}",
                kib, bytes, mt_index, mt_flags, v(w_ok), v(r_ok), r_first, fence_note
            );
        }

        println!();
        if any_fail {
            println!("RESULT: at least one size FAILED — GPU writes not visible to host at that size.");
            std::process::exit(2);
        } else {
            println!("RESULT: all sizes PASS — GPU compute writes reach the host buffer.");
        }
    }
}
