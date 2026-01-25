///
/// pygcode_viewer - Framebuffer Renderer Implementation
///

#include "framebuffer_renderer.hpp"

#include <glad/gl.h>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// STB Image Write for PNG and JPEG output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace pygcode {

//=============================================================================
// FramebufferRenderer
//=============================================================================

FramebufferRenderer::FramebufferRenderer() = default;

FramebufferRenderer::~FramebufferRenderer() {
    cleanup();
}

void FramebufferRenderer::cleanup() {
    if (m_fbo) {
        glDeleteFramebuffers(1, &m_fbo);
        m_fbo = 0;
    }
    if (m_color_texture) {
        glDeleteTextures(1, &m_color_texture);
        m_color_texture = 0;
    }
    if (m_depth_renderbuffer) {
        glDeleteRenderbuffers(1, &m_depth_renderbuffer);
        m_depth_renderbuffer = 0;
    }
    m_initialized = false;
}

bool FramebufferRenderer::init(int width, int height) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    
    cleanup();
    
    m_width = width;
    m_height = height;
    
    // Create framebuffer
    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    
    // Create color texture
    glGenTextures(1, &m_color_texture);
    glBindTexture(GL_TEXTURE_2D, m_color_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_color_texture, 0);
    
    // Create depth renderbuffer
    glGenRenderbuffers(1, &m_depth_renderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, m_depth_renderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_depth_renderbuffer);
    
    // Check framebuffer completeness
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        cleanup();
        return false;
    }
    
    // Unbind
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    
    m_initialized = true;
    return true;
}

bool FramebufferRenderer::resize(int width, int height) {
    if (width == m_width && height == m_height) {
        return true;
    }
    return init(width, height);
}

void FramebufferRenderer::bind() {
    if (m_initialized) {
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
        glViewport(0, 0, m_width, m_height);
    }
}

void FramebufferRenderer::unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FramebufferRenderer::clear(float r, float g, float b, float a) {
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void FramebufferRenderer::clear(const std::array<float, 4>& color) {
    clear(color[0], color[1], color[2], color[3]);
}

std::vector<uint8_t> FramebufferRenderer::read_pixels() const {
    if (!m_initialized) {
        return {};
    }
    
    std::vector<uint8_t> pixels(m_width * m_height * 4);
    
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Flip vertically (OpenGL origin is bottom-left)
    std::vector<uint8_t> flipped(pixels.size());
    int row_size = m_width * 4;
    for (int y = 0; y < m_height; ++y) {
        std::memcpy(
            flipped.data() + y * row_size,
            pixels.data() + (m_height - 1 - y) * row_size,
            row_size
        );
    }
    
    return flipped;
}

bool FramebufferRenderer::save_png(const std::string& filepath) const {
    return save_image(filepath);
}

bool FramebufferRenderer::save_image(const std::string& filepath, int jpeg_quality) const {
    if (!m_initialized) {
        return false;
    }
    
    auto pixels = read_pixels();
    if (pixels.empty()) {
        return false;
    }
    
    // Determine format from extension
    std::string ext = filepath.substr(filepath.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "jpg" || ext == "jpeg") {
        // Convert RGBA to RGB for JPEG (JPEG doesn't support alpha)
        std::vector<uint8_t> rgb_pixels(m_width * m_height * 3);
        for (int i = 0; i < m_width * m_height; ++i) {
            rgb_pixels[i * 3 + 0] = pixels[i * 4 + 0];
            rgb_pixels[i * 3 + 1] = pixels[i * 4 + 1];
            rgb_pixels[i * 3 + 2] = pixels[i * 4 + 2];
        }
        return stbi_write_jpg(filepath.c_str(), m_width, m_height, 3, rgb_pixels.data(), jpeg_quality) != 0;
    } else {
        // Default to PNG
        return stbi_write_png(filepath.c_str(), m_width, m_height, 4, pixels.data(), m_width * 4) != 0;
    }
}

bool FramebufferRenderer::render_to_file(const std::string& filepath,
                                          const std::array<float, 4>& bg_color,
                                          RenderCallback render_fn,
                                          int jpeg_quality) {
    if (!m_initialized) {
        return false;
    }
    
    bind();
    clear(bg_color);
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    render_fn();
    
    glFinish();
    
    bool result = save_image(filepath, jpeg_quality);
    
    unbind();
    
    return result;
}

//=============================================================================
// PNGWriter - Minimal PNG implementation using CRC and DEFLATE
// Uses a minimal uncompressed PNG format for simplicity
//=============================================================================

// CRC32 table
static uint32_t crc_table[256];
static bool crc_table_computed = false;

static void make_crc_table() {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            if (c & 1) {
                c = 0xedb88320L ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        crc_table[n] = c;
    }
    crc_table_computed = true;
}

static uint32_t crc32(const uint8_t* data, size_t length) {
    if (!crc_table_computed) {
        make_crc_table();
    }
    
    uint32_t c = 0xffffffffL;
    for (size_t n = 0; n < length; n++) {
        c = crc_table[(c ^ data[n]) & 0xff] ^ (c >> 8);
    }
    return c ^ 0xffffffffL;
}

static void write_be32(std::vector<uint8_t>& out, uint32_t val) {
    out.push_back((val >> 24) & 0xff);
    out.push_back((val >> 16) & 0xff);
    out.push_back((val >> 8) & 0xff);
    out.push_back(val & 0xff);
}

static void write_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    // Length
    write_be32(out, static_cast<uint32_t>(data.size()));
    
    // Type + Data for CRC
    std::vector<uint8_t> crc_data;
    crc_data.insert(crc_data.end(), type, type + 4);
    crc_data.insert(crc_data.end(), data.begin(), data.end());
    
    // Type
    out.insert(out.end(), type, type + 4);
    
    // Data
    out.insert(out.end(), data.begin(), data.end());
    
    // CRC
    write_be32(out, crc32(crc_data.data(), crc_data.size()));
}

// Adler32 for zlib
static uint32_t adler32(const uint8_t* data, size_t length) {
    uint32_t a = 1, b = 0;
    const uint32_t MOD = 65521;
    
    for (size_t i = 0; i < length; ++i) {
        a = (a + data[i]) % MOD;
        b = (b + a) % MOD;
    }
    
    return (b << 16) | a;
}

bool PNGWriter::write(const std::string& filepath, int width, int height, const uint8_t* pixels) {
    std::vector<uint8_t> png_data;
    
    // PNG signature
    const uint8_t signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    png_data.insert(png_data.end(), signature, signature + 8);
    
    // IHDR chunk
    {
        std::vector<uint8_t> ihdr;
        write_be32(ihdr, width);
        write_be32(ihdr, height);
        ihdr.push_back(8);  // Bit depth
        ihdr.push_back(6);  // Color type: RGBA
        ihdr.push_back(0);  // Compression method
        ihdr.push_back(0);  // Filter method
        ihdr.push_back(0);  // Interlace method
        write_chunk(png_data, "IHDR", ihdr);
    }
    
    // IDAT chunk (image data with zlib compression)
    {
        // Prepare raw image data with filter bytes
        std::vector<uint8_t> raw_data;
        int row_size = width * 4;
        for (int y = 0; y < height; ++y) {
            raw_data.push_back(0);  // Filter type: None
            raw_data.insert(raw_data.end(), 
                           pixels + y * row_size,
                           pixels + (y + 1) * row_size);
        }
        
        // Create uncompressed zlib stream (store method)
        std::vector<uint8_t> zlib_data;
        
        // Zlib header: CMF=0x78 (deflate, 32K window), FLG=0x01 (no dict, check bits)
        zlib_data.push_back(0x78);
        zlib_data.push_back(0x01);
        
        // DEFLATE stored blocks
        size_t pos = 0;
        while (pos < raw_data.size()) {
            size_t chunk_size = std::min(raw_data.size() - pos, size_t(65535));
            bool is_last = (pos + chunk_size >= raw_data.size());
            
            // Block header: BFINAL (1 bit) + BTYPE=00 (2 bits, stored)
            zlib_data.push_back(is_last ? 0x01 : 0x00);
            
            // LEN and NLEN
            uint16_t len = static_cast<uint16_t>(chunk_size);
            uint16_t nlen = ~len;
            zlib_data.push_back(len & 0xff);
            zlib_data.push_back((len >> 8) & 0xff);
            zlib_data.push_back(nlen & 0xff);
            zlib_data.push_back((nlen >> 8) & 0xff);
            
            // Data
            zlib_data.insert(zlib_data.end(), 
                            raw_data.begin() + pos,
                            raw_data.begin() + pos + chunk_size);
            
            pos += chunk_size;
        }
        
        // Adler32 checksum
        uint32_t adler = adler32(raw_data.data(), raw_data.size());
        zlib_data.push_back((adler >> 24) & 0xff);
        zlib_data.push_back((adler >> 16) & 0xff);
        zlib_data.push_back((adler >> 8) & 0xff);
        zlib_data.push_back(adler & 0xff);
        
        write_chunk(png_data, "IDAT", zlib_data);
    }
    
    // IEND chunk
    write_chunk(png_data, "IEND", {});
    
    // Write to file
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(png_data.data()), png_data.size());
    return file.good();
}

} // namespace pygcode
