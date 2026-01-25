///
/// pygcode_viewer - Config Parser
/// JSON configuration parser mapping to libvgcode settings
///

#ifndef PYGCODE_CONFIG_PARSER_HPP
#define PYGCODE_CONFIG_PARSER_HPP

#include <string>
#include <vector>
#include <map>
#include <array>
#include <cstdint>

#include "Types.hpp"

namespace libvgcode {
class Viewer;
}

namespace pygcode {

/// Parsed view configuration
struct ViewConfig {
    // View type
    libvgcode::EViewType view_type = libvgcode::EViewType::FeatureType;
    
    // Layer range (0 = use full range)
    size_t layer_min = 0;
    size_t layer_max = 0;  // 0 means max available
    bool has_layer_range = false;
    
    // Feature visibility
    std::map<libvgcode::EOptionType, bool> option_visibility;
    
    // Extrusion role visibility and colors
    std::map<libvgcode::EGCodeExtrusionRole, bool> role_visibility;
    std::map<libvgcode::EGCodeExtrusionRole, libvgcode::Color> role_colors;
    
    // Tool colors
    std::vector<libvgcode::Color> tool_colors;
    
    // Output settings
    int width = 1920;
    int height = 1080;
    std::array<float, 4> background_color = {1.0f, 1.0f, 1.0f, 1.0f};  // RGBA
    
    // Initialize with defaults
    void init_defaults();
};

/// Configuration parser
class ConfigParser {
public:
    /// Parse JSON string into ViewConfig
    static ViewConfig parse_json(const std::string& json_str);
    
    /// Parse JSON file into ViewConfig
    static ViewConfig parse_file(const std::string& filepath);
    
    /// Apply configuration to a libvgcode::Viewer
    static void apply_config(libvgcode::Viewer& viewer, const ViewConfig& config);
    
    /// Convert view type string to enum
    static libvgcode::EViewType parse_view_type(const std::string& type_str);
    
    /// Convert option type string to enum
    static libvgcode::EOptionType parse_option_type(const std::string& type_str);
    
    /// Convert extrusion role string to enum
    static libvgcode::EGCodeExtrusionRole parse_extrusion_role(const std::string& role_str);
    
    /// Parse hex color string (#RRGGBB) to Color
    static libvgcode::Color parse_color(const std::string& color_str);
    
    /// Parse RGBA color array [r, g, b, a] to float array
    static std::array<float, 4> parse_rgba(const std::string& color_str);
};

/// Simple JSON value (minimal implementation)
class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    
    JsonValue() : m_type(Type::Null) {}
    JsonValue(bool b) : m_type(Type::Bool), m_bool(b) {}
    JsonValue(double n) : m_type(Type::Number), m_number(n) {}
    JsonValue(const std::string& s) : m_type(Type::String), m_string(s) {}
    JsonValue(std::vector<JsonValue> arr) : m_type(Type::Array), m_array(std::move(arr)) {}
    JsonValue(std::map<std::string, JsonValue> obj) : m_type(Type::Object), m_object(std::move(obj)) {}
    
    Type type() const { return m_type; }
    bool is_null() const { return m_type == Type::Null; }
    bool is_bool() const { return m_type == Type::Bool; }
    bool is_number() const { return m_type == Type::Number; }
    bool is_string() const { return m_type == Type::String; }
    bool is_array() const { return m_type == Type::Array; }
    bool is_object() const { return m_type == Type::Object; }
    
    bool as_bool() const { return m_bool; }
    double as_number() const { return m_number; }
    int as_int() const { return static_cast<int>(m_number); }
    const std::string& as_string() const { return m_string; }
    const std::vector<JsonValue>& as_array() const { return m_array; }
    const std::map<std::string, JsonValue>& as_object() const { return m_object; }
    
    bool has(const std::string& key) const {
        return m_type == Type::Object && m_object.find(key) != m_object.end();
    }
    
    const JsonValue& operator[](const std::string& key) const {
        static JsonValue null_value;
        if (m_type != Type::Object) return null_value;
        auto it = m_object.find(key);
        return it != m_object.end() ? it->second : null_value;
    }
    
    const JsonValue& operator[](size_t index) const {
        static JsonValue null_value;
        if (m_type != Type::Array || index >= m_array.size()) return null_value;
        return m_array[index];
    }
    
    size_t size() const {
        if (m_type == Type::Array) return m_array.size();
        if (m_type == Type::Object) return m_object.size();
        return 0;
    }
    
    /// Parse JSON string
    static JsonValue parse(const std::string& json_str);
    
private:
    Type m_type;
    bool m_bool = false;
    double m_number = 0.0;
    std::string m_string;
    std::vector<JsonValue> m_array;
    std::map<std::string, JsonValue> m_object;
};

} // namespace pygcode

#endif // PYGCODE_CONFIG_PARSER_HPP
