///
/// pygcode_viewer - Headless OpenGL Context Implementation
///

#include "headless_context.hpp"

#include <stdexcept>
#include <cstring>

// Include GLAD for OpenGL
#include <glad/gl.h>

#ifdef PYGCODE_PLATFORM_LINUX
#include <EGL/egl.h>
#include <EGL/eglext.h>
#ifdef HAVE_GBM
#include <gbm.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#endif

#ifdef PYGCODE_PLATFORM_MACOS
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <dlfcn.h>

// macOS doesn't have CGLGetProcAddress, use dlsym instead
static void* macos_get_proc_address(const char* name) {
    static void* lib = nullptr;
    if (!lib) {
        lib = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_LAZY);
    }
    return lib ? dlsym(lib, name) : nullptr;
}
#endif

#ifdef PYGCODE_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <gl/GL.h>
#endif

namespace pygcode {

//=============================================================================
// Factory method
//=============================================================================

std::unique_ptr<HeadlessContext> HeadlessContext::create_context() {
#ifdef PYGCODE_PLATFORM_LINUX
    return std::make_unique<EGLHeadlessContext>();
#elif defined(PYGCODE_PLATFORM_MACOS)
    return std::make_unique<CGLHeadlessContext>();
#elif defined(PYGCODE_PLATFORM_WINDOWS)
    return std::make_unique<WGLHeadlessContext>();
#else
    throw std::runtime_error("No headless context implementation for this platform");
#endif
}

//=============================================================================
// Linux EGL Implementation
//=============================================================================

#ifdef PYGCODE_PLATFORM_LINUX

struct EGLHeadlessContext::Impl {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    EGLConfig config = nullptr;
    bool valid = false;
    std::string version_string;
    
#ifdef HAVE_GBM
    int drm_fd = -1;
    struct gbm_device* gbm = nullptr;
    struct gbm_surface* gbm_surf = nullptr;
#endif
};

EGLHeadlessContext::EGLHeadlessContext() : m_impl(std::make_unique<Impl>()) {}

EGLHeadlessContext::~EGLHeadlessContext() {
    release();
}

bool EGLHeadlessContext::create(int width, int height) {
    // Try surfaceless first (most compatible for headless)
    static const EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    };
    
    static const EGLint pbuffer_attribs[] = {
        EGL_WIDTH, width,
        EGL_HEIGHT, height,
        EGL_NONE
    };
    
    static const EGLint context_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    
    // Get display
    m_impl->display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (m_impl->display == EGL_NO_DISPLAY) {
        // Try platform display
        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = 
            (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
        if (eglGetPlatformDisplayEXT) {
            m_impl->display = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        }
    }
    
    if (m_impl->display == EGL_NO_DISPLAY) {
        return false;
    }
    
    // Initialize EGL
    EGLint major, minor;
    if (!eglInitialize(m_impl->display, &major, &minor)) {
        return false;
    }
    
    // Bind OpenGL API
    if (!eglBindAPI(EGL_OPENGL_API)) {
        return false;
    }
    
    // Choose config
    EGLint num_configs;
    if (!eglChooseConfig(m_impl->display, config_attribs, &m_impl->config, 1, &num_configs) || num_configs == 0) {
        return false;
    }
    
    // Create pbuffer surface
    m_impl->surface = eglCreatePbufferSurface(m_impl->display, m_impl->config, pbuffer_attribs);
    if (m_impl->surface == EGL_NO_SURFACE) {
        // Try without surface (surfaceless rendering)
        m_impl->surface = EGL_NO_SURFACE;
    }
    
    // Create context
    m_impl->context = eglCreateContext(m_impl->display, m_impl->config, EGL_NO_CONTEXT, context_attribs);
    if (m_impl->context == EGL_NO_CONTEXT) {
        // Try with older OpenGL version
        static const EGLint context_attribs_compat[] = {
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 2,
            EGL_NONE
        };
        m_impl->context = eglCreateContext(m_impl->display, m_impl->config, EGL_NO_CONTEXT, context_attribs_compat);
    }
    
    if (m_impl->context == EGL_NO_CONTEXT) {
        return false;
    }
    
    // Make current
    if (!make_current()) {
        return false;
    }
    
    // Load OpenGL functions with GLAD
    if (!gladLoadGL((GLADloadfunc)eglGetProcAddress)) {
        return false;
    }
    
    m_impl->version_string = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    m_impl->valid = true;
    
    return true;
}

bool EGLHeadlessContext::make_current() {
    if (m_impl->display == EGL_NO_DISPLAY || m_impl->context == EGL_NO_CONTEXT) {
        return false;
    }
    return eglMakeCurrent(m_impl->display, m_impl->surface, m_impl->surface, m_impl->context) == EGL_TRUE;
}

void EGLHeadlessContext::release() {
    if (m_impl->display != EGL_NO_DISPLAY) {
        eglMakeCurrent(m_impl->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        
        if (m_impl->context != EGL_NO_CONTEXT) {
            eglDestroyContext(m_impl->display, m_impl->context);
            m_impl->context = EGL_NO_CONTEXT;
        }
        
        if (m_impl->surface != EGL_NO_SURFACE) {
            eglDestroySurface(m_impl->display, m_impl->surface);
            m_impl->surface = EGL_NO_SURFACE;
        }
        
        eglTerminate(m_impl->display);
        m_impl->display = EGL_NO_DISPLAY;
    }
    
#ifdef HAVE_GBM
    if (m_impl->gbm_surf) {
        gbm_surface_destroy(m_impl->gbm_surf);
        m_impl->gbm_surf = nullptr;
    }
    if (m_impl->gbm) {
        gbm_device_destroy(m_impl->gbm);
        m_impl->gbm = nullptr;
    }
    if (m_impl->drm_fd >= 0) {
        close(m_impl->drm_fd);
        m_impl->drm_fd = -1;
    }
#endif
    
    m_impl->valid = false;
}

void EGLHeadlessContext::swap_buffers() {
    if (m_impl->surface != EGL_NO_SURFACE) {
        eglSwapBuffers(m_impl->display, m_impl->surface);
    }
}

std::string EGLHeadlessContext::get_version_string() const {
    return m_impl->version_string;
}

bool EGLHeadlessContext::is_valid() const {
    return m_impl->valid;
}

#endif // PYGCODE_PLATFORM_LINUX

//=============================================================================
// macOS CGL Implementation
//=============================================================================

#ifdef PYGCODE_PLATFORM_MACOS

struct CGLHeadlessContext::Impl {
    CGLContextObj context = nullptr;
    CGLPixelFormatObj pixel_format = nullptr;
    bool valid = false;
    std::string version_string;
};

CGLHeadlessContext::CGLHeadlessContext() : m_impl(std::make_unique<Impl>()) {}

CGLHeadlessContext::~CGLHeadlessContext() {
    release();
}

bool CGLHeadlessContext::create(int width, int height) {
    (void)width;
    (void)height;
    
    // Pixel format attributes for OpenGL 3.2+ Core Profile
    CGLPixelFormatAttribute attribs[] = {
        kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)kCGLOGLPVersion_3_2_Core,
        kCGLPFAColorSize, (CGLPixelFormatAttribute)24,
        kCGLPFAAlphaSize, (CGLPixelFormatAttribute)8,
        kCGLPFADepthSize, (CGLPixelFormatAttribute)24,
        kCGLPFAAccelerated,
        kCGLPFAAllowOfflineRenderers,
        (CGLPixelFormatAttribute)0
    };
    
    GLint num_pixel_formats;
    CGLError err = CGLChoosePixelFormat(attribs, &m_impl->pixel_format, &num_pixel_formats);
    if (err != kCGLNoError || m_impl->pixel_format == nullptr) {
        return false;
    }
    
    err = CGLCreateContext(m_impl->pixel_format, nullptr, &m_impl->context);
    if (err != kCGLNoError || m_impl->context == nullptr) {
        CGLDestroyPixelFormat(m_impl->pixel_format);
        m_impl->pixel_format = nullptr;
        return false;
    }
    
    if (!make_current()) {
        release();
        return false;
    }
    
    // Load OpenGL functions with GLAD
    if (!gladLoadGL((GLADloadfunc)macos_get_proc_address)) {
        release();
        return false;
    }
    
    m_impl->version_string = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    m_impl->valid = true;
    
    return true;
}

bool CGLHeadlessContext::make_current() {
    if (m_impl->context == nullptr) {
        return false;
    }
    return CGLSetCurrentContext(m_impl->context) == kCGLNoError;
}

void CGLHeadlessContext::release() {
    if (m_impl->context != nullptr) {
        CGLSetCurrentContext(nullptr);
        CGLDestroyContext(m_impl->context);
        m_impl->context = nullptr;
    }
    if (m_impl->pixel_format != nullptr) {
        CGLDestroyPixelFormat(m_impl->pixel_format);
        m_impl->pixel_format = nullptr;
    }
    m_impl->valid = false;
}

void CGLHeadlessContext::swap_buffers() {
    if (m_impl->context != nullptr) {
        CGLFlushDrawable(m_impl->context);
    }
}

std::string CGLHeadlessContext::get_version_string() const {
    return m_impl->version_string;
}

bool CGLHeadlessContext::is_valid() const {
    return m_impl->valid;
}

#endif // PYGCODE_PLATFORM_MACOS

//=============================================================================
// Windows WGL Implementation
//=============================================================================

#ifdef PYGCODE_PLATFORM_WINDOWS

struct WGLHeadlessContext::Impl {
    HWND hwnd = nullptr;
    HDC hdc = nullptr;
    HGLRC hglrc = nullptr;
    bool valid = false;
    std::string version_string;
};

WGLHeadlessContext::WGLHeadlessContext() : m_impl(std::make_unique<Impl>()) {}

WGLHeadlessContext::~WGLHeadlessContext() {
    release();
}

bool WGLHeadlessContext::create(int width, int height) {
    // Register window class
    WNDCLASSA wc = {};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "PygcodeViewerHidden";
    RegisterClassA(&wc);
    
    // Create hidden window
    m_impl->hwnd = CreateWindowExA(
        0, wc.lpszClassName, "Hidden",
        WS_OVERLAPPEDWINDOW,
        0, 0, width, height,
        nullptr, nullptr, wc.hInstance, nullptr
    );
    
    if (!m_impl->hwnd) {
        return false;
    }
    
    m_impl->hdc = GetDC(m_impl->hwnd);
    if (!m_impl->hdc) {
        DestroyWindow(m_impl->hwnd);
        m_impl->hwnd = nullptr;
        return false;
    }
    
    // Set pixel format
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;
    
    int format = ChoosePixelFormat(m_impl->hdc, &pfd);
    if (!format || !SetPixelFormat(m_impl->hdc, format, &pfd)) {
        ReleaseDC(m_impl->hwnd, m_impl->hdc);
        DestroyWindow(m_impl->hwnd);
        m_impl->hwnd = nullptr;
        m_impl->hdc = nullptr;
        return false;
    }
    
    // Create basic context first
    m_impl->hglrc = wglCreateContext(m_impl->hdc);
    if (!m_impl->hglrc) {
        ReleaseDC(m_impl->hwnd, m_impl->hdc);
        DestroyWindow(m_impl->hwnd);
        m_impl->hwnd = nullptr;
        m_impl->hdc = nullptr;
        return false;
    }
    
    if (!make_current()) {
        release();
        return false;
    }
    
    // Load OpenGL functions with GLAD
    if (!gladLoadGL((GLADloadfunc)wglGetProcAddress)) {
        release();
        return false;
    }
    
    m_impl->version_string = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    m_impl->valid = true;
    
    return true;
}

bool WGLHeadlessContext::make_current() {
    if (!m_impl->hglrc || !m_impl->hdc) {
        return false;
    }
    return wglMakeCurrent(m_impl->hdc, m_impl->hglrc) == TRUE;
}

void WGLHeadlessContext::release() {
    if (m_impl->hglrc) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(m_impl->hglrc);
        m_impl->hglrc = nullptr;
    }
    if (m_impl->hdc && m_impl->hwnd) {
        ReleaseDC(m_impl->hwnd, m_impl->hdc);
        m_impl->hdc = nullptr;
    }
    if (m_impl->hwnd) {
        DestroyWindow(m_impl->hwnd);
        m_impl->hwnd = nullptr;
    }
    m_impl->valid = false;
}

void WGLHeadlessContext::swap_buffers() {
    if (m_impl->hdc) {
        SwapBuffers(m_impl->hdc);
    }
}

std::string WGLHeadlessContext::get_version_string() const {
    return m_impl->version_string;
}

bool WGLHeadlessContext::is_valid() const {
    return m_impl->valid;
}

#endif // PYGCODE_PLATFORM_WINDOWS

} // namespace pygcode
