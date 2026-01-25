///
/// pygcode_viewer - Framebuffer Renderer
/// FBO-based offscreen rendering and PNG export
///

#ifndef PYGCODE_FRAMEBUFFER_RENDERER_HPP
#define PYGCODE_FRAMEBUFFER_RENDERER_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include <functional>

namespace pygcode {

/// Framebuffer for offscreen rendering
class FramebufferRenderer {
public:
    FramebufferRenderer();
    ~FramebufferRenderer();
    
    // No copy
    FramebufferRenderer(const FramebufferRenderer&) = delete;
    FramebufferRenderer& operator=(const FramebufferRenderer&) = delete;
    
    /// Initialize the framebuffer with given dimensions
    bool init(int width, int height);
    
    /// Resize the framebuffer
    bool resize(int width, int height);
    
    /// Get framebuffer dimensions
    int get_width() const { return m_width; }
    int get_height() const { return m_height; }
    
    /// Bind the framebuffer for rendering
    void bind();
    
    /// Unbind the framebuffer (bind default framebuffer)
    void unbind();
    
    /// Clear the framebuffer with given color
    void clear(float r, float g, float b, float a = 1.0f);
    void clear(const std::array<float, 4>& color);
    
    /// Read pixels from the framebuffer
    std::vector<uint8_t> read_pixels() const;
    
    /// Save framebuffer contents to PNG file
    bool save_png(const std::string& filepath) const;
    
    /// Save framebuffer contents to image file (PNG or JPEG based on extension)
    /// jpeg_quality: 1-100 for JPEG (default 90)
    bool save_image(const std::string& filepath, int jpeg_quality = 90) const;
    
    /// Render callback type
    using RenderCallback = std::function<void()>;
    
    /// Render using callback and save to file (format based on extension)
    bool render_to_file(const std::string& filepath, 
                        const std::array<float, 4>& bg_color,
                        RenderCallback render_fn,
                        int jpeg_quality = 90);
    
private:
    void cleanup();
    
    int m_width = 0;
    int m_height = 0;
    
    unsigned int m_fbo = 0;
    unsigned int m_color_texture = 0;
    unsigned int m_depth_renderbuffer = 0;
    
    bool m_initialized = false;
};

/// Simple PNG writer (no external dependencies)
class PNGWriter {
public:
    /// Write RGBA pixel data to PNG file
    static bool write(const std::string& filepath, 
                      int width, int height,
                      const uint8_t* pixels);
};

} // namespace pygcode

#endif // PYGCODE_FRAMEBUFFER_RENDERER_HPP
