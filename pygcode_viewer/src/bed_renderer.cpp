///
/// pygcode_viewer - Bed Renderer Implementation
///

#include "bed_renderer.hpp"
#include <glad/gl.h>
#include <vector>
#include <cmath>

namespace pygcode {

// Simple vertex shader for bed lines
static const char* bed_vertex_shader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
}
)";

// Simple fragment shader for bed lines
static const char* bed_fragment_shader = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 lineColor;
void main() {
    FragColor = lineColor;
}
)";

BedRenderer::BedRenderer() = default;

BedRenderer::~BedRenderer() {
    cleanup();
}

void BedRenderer::cleanup() {
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    if (m_shader_program) {
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
    }
    m_initialized = false;
}

static unsigned int compile_shader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool BedRenderer::init() {
    if (m_initialized) return true;
    
    // Compile shaders
    unsigned int vertex_shader = compile_shader(GL_VERTEX_SHADER, bed_vertex_shader);
    if (!vertex_shader) return false;
    
    unsigned int fragment_shader = compile_shader(GL_FRAGMENT_SHADER, bed_fragment_shader);
    if (!fragment_shader) {
        glDeleteShader(vertex_shader);
        return false;
    }
    
    // Link program
    m_shader_program = glCreateProgram();
    glAttachShader(m_shader_program, vertex_shader);
    glAttachShader(m_shader_program, fragment_shader);
    glLinkProgram(m_shader_program);
    
    int success;
    glGetProgramiv(m_shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        glDeleteProgram(m_shader_program);
        m_shader_program = 0;
        return false;
    }
    
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    
    // Create VAO and VBO
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    
    m_initialized = true;
    return true;
}

void BedRenderer::set_config(const BedConfig& config) {
    m_config = config;
}

void BedRenderer::render(const float* view_matrix, const float* projection_matrix) {
    if (!m_initialized) return;
    
    if (m_config.show_outline) {
        render_outline(view_matrix, projection_matrix);
    }
    
    if (m_config.show_grid) {
        render_grid(view_matrix, projection_matrix);
    }
}

void BedRenderer::render_outline(const float* view_matrix, const float* projection_matrix) {
    // Calculate bed corners
    float x0 = -m_config.origin_x;
    float y0 = -m_config.origin_y;
    float x1 = m_config.size_x - m_config.origin_x;
    float y1 = m_config.size_y - m_config.origin_y;
    float z = m_config.z_position;
    
    // Bed outline vertices (closed rectangle)
    float vertices[] = {
        x0, y0, z,
        x1, y0, z,
        x1, y1, z,
        x0, y1, z,
        x0, y0, z  // Close the loop
    };
    
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glUseProgram(m_shader_program);
    
    // Set uniforms
    int view_loc = glGetUniformLocation(m_shader_program, "view");
    int proj_loc = glGetUniformLocation(m_shader_program, "projection");
    int color_loc = glGetUniformLocation(m_shader_program, "lineColor");
    
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix);
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix);
    glUniform4fv(color_loc, 1, m_config.outline_color.data());
    
    // Draw
    glLineWidth(m_config.line_width);
    glDrawArrays(GL_LINE_STRIP, 0, 5);
    
    glBindVertexArray(0);
    glUseProgram(0);
}

void BedRenderer::render_grid(const float* view_matrix, const float* projection_matrix) {
    // Calculate bed bounds
    float x0 = -m_config.origin_x;
    float y0 = -m_config.origin_y;
    float x1 = m_config.size_x - m_config.origin_x;
    float y1 = m_config.size_y - m_config.origin_y;
    float z = m_config.z_position;
    float spacing = m_config.grid_spacing;
    
    // Generate grid line vertices
    std::vector<float> vertices;
    
    // Vertical lines
    for (float x = x0; x <= x1; x += spacing) {
        vertices.push_back(x); vertices.push_back(y0); vertices.push_back(z);
        vertices.push_back(x); vertices.push_back(y1); vertices.push_back(z);
    }
    
    // Horizontal lines
    for (float y = y0; y <= y1; y += spacing) {
        vertices.push_back(x0); vertices.push_back(y); vertices.push_back(z);
        vertices.push_back(x1); vertices.push_back(y); vertices.push_back(z);
    }
    
    if (vertices.empty()) return;
    
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glUseProgram(m_shader_program);
    
    // Set uniforms
    int view_loc = glGetUniformLocation(m_shader_program, "view");
    int proj_loc = glGetUniformLocation(m_shader_program, "projection");
    int color_loc = glGetUniformLocation(m_shader_program, "lineColor");
    
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix);
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix);
    glUniform4fv(color_loc, 1, m_config.grid_color.data());
    
    // Draw
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, static_cast<int>(vertices.size() / 3));
    
    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace pygcode
