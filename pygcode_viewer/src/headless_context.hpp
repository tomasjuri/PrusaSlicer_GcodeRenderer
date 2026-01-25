///
/// pygcode_viewer - Headless OpenGL Context
/// Platform-specific headless OpenGL context creation
///

#ifndef PYGCODE_HEADLESS_CONTEXT_HPP
#define PYGCODE_HEADLESS_CONTEXT_HPP

#include <string>
#include <memory>

namespace pygcode {

/// Abstract base class for headless OpenGL context
class HeadlessContext {
public:
    virtual ~HeadlessContext() = default;
    
    /// Create and make current the OpenGL context
    virtual bool create(int width, int height) = 0;
    
    /// Make this context current
    virtual bool make_current() = 0;
    
    /// Release the context
    virtual void release() = 0;
    
    /// Swap buffers (for double-buffered contexts)
    virtual void swap_buffers() = 0;
    
    /// Get OpenGL version string (after context is created)
    virtual std::string get_version_string() const = 0;
    
    /// Check if context is valid
    virtual bool is_valid() const = 0;
    
    /// Factory method to create platform-specific context
    static std::unique_ptr<HeadlessContext> create_context();
};

#ifdef PYGCODE_PLATFORM_LINUX

/// EGL-based headless context for Linux
class EGLHeadlessContext : public HeadlessContext {
public:
    EGLHeadlessContext();
    ~EGLHeadlessContext() override;
    
    bool create(int width, int height) override;
    bool make_current() override;
    void release() override;
    void swap_buffers() override;
    std::string get_version_string() const override;
    bool is_valid() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // PYGCODE_PLATFORM_LINUX

#ifdef PYGCODE_PLATFORM_MACOS

/// CGL-based headless context for macOS
class CGLHeadlessContext : public HeadlessContext {
public:
    CGLHeadlessContext();
    ~CGLHeadlessContext() override;
    
    bool create(int width, int height) override;
    bool make_current() override;
    void release() override;
    void swap_buffers() override;
    std::string get_version_string() const override;
    bool is_valid() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // PYGCODE_PLATFORM_MACOS

#ifdef PYGCODE_PLATFORM_WINDOWS

/// WGL-based headless context for Windows
class WGLHeadlessContext : public HeadlessContext {
public:
    WGLHeadlessContext();
    ~WGLHeadlessContext() override;
    
    bool create(int width, int height) override;
    bool make_current() override;
    void release() override;
    void swap_buffers() override;
    std::string get_version_string() const override;
    bool is_valid() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // PYGCODE_PLATFORM_WINDOWS

} // namespace pygcode

#endif // PYGCODE_HEADLESS_CONTEXT_HPP
