///
/// pygcode_viewer - Config Parser Implementation
///

#include "config_parser.hpp"
#include "Viewer.hpp"

#include <fstream>
#include <sstream>
#include <cctype>
#include <stdexcept>
#include <algorithm>

namespace pygcode {

//=============================================================================
// ViewConfig
//=============================================================================

void ViewConfig::init_defaults() {
    view_type = libvgcode::EViewType::FeatureType;
    layer_min = 0;
    layer_max = 0;
    has_layer_range = false;
    
    // Default option visibility - only extrusions visible
    option_visibility[libvgcode::EOptionType::Travels] = false;
    option_visibility[libvgcode::EOptionType::Wipes] = false;
    option_visibility[libvgcode::EOptionType::Retractions] = false;
    option_visibility[libvgcode::EOptionType::Unretractions] = false;
    option_visibility[libvgcode::EOptionType::Seams] = false;
    option_visibility[libvgcode::EOptionType::ToolChanges] = false;
    option_visibility[libvgcode::EOptionType::ColorChanges] = false;
    option_visibility[libvgcode::EOptionType::PausePrints] = false;
    option_visibility[libvgcode::EOptionType::CustomGCodes] = false;
    
    // Default role visibility (all visible)
    for (int i = 0; i < static_cast<int>(libvgcode::EGCodeExtrusionRole::COUNT); ++i) {
        auto role = static_cast<libvgcode::EGCodeExtrusionRole>(i);
        role_visibility[role] = true;
    }
    
    // Default tool colors
    tool_colors = {
        {255, 128, 0},   // Orange
        {0, 255, 0},     // Green
        {0, 0, 255},     // Blue
        {255, 0, 255},   // Magenta
        {0, 255, 255},   // Cyan
    };
    
    // Output defaults
    width = 1920;
    height = 1080;
    // Dark gray background (almost black) - configurable
    background_color = {0.1f, 0.1f, 0.1f, 1.0f};
    
    // Set exact PrusaSlicer default role colors
    role_colors[libvgcode::EGCodeExtrusionRole::None]                     = {230, 179, 179};
    role_colors[libvgcode::EGCodeExtrusionRole::Perimeter]                = {255, 230,  77};
    role_colors[libvgcode::EGCodeExtrusionRole::ExternalPerimeter]        = {255, 125,  56};
    role_colors[libvgcode::EGCodeExtrusionRole::OverhangPerimeter]        = { 31,  31, 255};
    role_colors[libvgcode::EGCodeExtrusionRole::InternalInfill]           = {176,  48,  41};
    role_colors[libvgcode::EGCodeExtrusionRole::SolidInfill]              = {150,  84, 204};
    role_colors[libvgcode::EGCodeExtrusionRole::TopSolidInfill]           = {240,  64,  64};
    role_colors[libvgcode::EGCodeExtrusionRole::Ironing]                  = {255, 140, 105};
    role_colors[libvgcode::EGCodeExtrusionRole::BridgeInfill]             = { 77, 128, 186};
    role_colors[libvgcode::EGCodeExtrusionRole::GapFill]                  = {255, 255, 255};
    role_colors[libvgcode::EGCodeExtrusionRole::Skirt]                    = {  0, 135, 110};
    role_colors[libvgcode::EGCodeExtrusionRole::SupportMaterial]          = {  0, 255,   0};
    role_colors[libvgcode::EGCodeExtrusionRole::SupportMaterialInterface] = {  0, 128,   0};
    role_colors[libvgcode::EGCodeExtrusionRole::WipeTower]                = {179, 227, 171};
    role_colors[libvgcode::EGCodeExtrusionRole::Custom]                   = { 94, 209, 148};
}

//=============================================================================
// JsonValue - Simple JSON Parser
//=============================================================================

class JsonParser {
public:
    explicit JsonParser(const std::string& str) : m_str(str), m_pos(0) {}
    
    JsonValue parse() {
        skip_whitespace();
        return parse_value();
    }
    
private:
    const std::string& m_str;
    size_t m_pos;
    
    char peek() const {
        return m_pos < m_str.size() ? m_str[m_pos] : '\0';
    }
    
    char get() {
        return m_pos < m_str.size() ? m_str[m_pos++] : '\0';
    }
    
    void skip_whitespace() {
        while (m_pos < m_str.size() && std::isspace(m_str[m_pos])) {
            ++m_pos;
        }
    }
    
    JsonValue parse_value() {
        skip_whitespace();
        char c = peek();
        
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(c)) return parse_number();
        
        throw std::runtime_error("Invalid JSON at position " + std::to_string(m_pos));
    }
    
    JsonValue parse_string() {
        get(); // consume opening quote
        std::string result;
        
        while (m_pos < m_str.size()) {
            char c = get();
            if (c == '"') break;
            if (c == '\\') {
                char escaped = get();
                switch (escaped) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: result += escaped; break;
                }
            } else {
                result += c;
            }
        }
        
        return JsonValue(result);
    }
    
    JsonValue parse_number() {
        size_t start = m_pos;
        
        if (peek() == '-') get();
        
        while (std::isdigit(peek())) get();
        
        if (peek() == '.') {
            get();
            while (std::isdigit(peek())) get();
        }
        
        if (peek() == 'e' || peek() == 'E') {
            get();
            if (peek() == '+' || peek() == '-') get();
            while (std::isdigit(peek())) get();
        }
        
        std::string num_str = m_str.substr(start, m_pos - start);
        return JsonValue(std::stod(num_str));
    }
    
    JsonValue parse_bool() {
        if (m_str.compare(m_pos, 4, "true") == 0) {
            m_pos += 4;
            return JsonValue(true);
        }
        if (m_str.compare(m_pos, 5, "false") == 0) {
            m_pos += 5;
            return JsonValue(false);
        }
        throw std::runtime_error("Invalid boolean");
    }
    
    JsonValue parse_null() {
        if (m_str.compare(m_pos, 4, "null") == 0) {
            m_pos += 4;
            return JsonValue();
        }
        throw std::runtime_error("Invalid null");
    }
    
    JsonValue parse_array() {
        get(); // consume '['
        std::vector<JsonValue> arr;
        
        skip_whitespace();
        if (peek() == ']') {
            get();
            return JsonValue(arr);
        }
        
        while (true) {
            arr.push_back(parse_value());
            skip_whitespace();
            
            if (peek() == ']') {
                get();
                break;
            }
            if (peek() == ',') {
                get();
            } else {
                throw std::runtime_error("Expected ',' or ']'");
            }
        }
        
        return JsonValue(arr);
    }
    
    JsonValue parse_object() {
        get(); // consume '{'
        std::map<std::string, JsonValue> obj;
        
        skip_whitespace();
        if (peek() == '}') {
            get();
            return JsonValue(obj);
        }
        
        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                throw std::runtime_error("Expected string key");
            }
            
            std::string key = parse_string().as_string();
            
            skip_whitespace();
            if (get() != ':') {
                throw std::runtime_error("Expected ':'");
            }
            
            obj[key] = parse_value();
            skip_whitespace();
            
            if (peek() == '}') {
                get();
                break;
            }
            if (peek() == ',') {
                get();
            } else {
                throw std::runtime_error("Expected ',' or '}'");
            }
        }
        
        return JsonValue(obj);
    }
};

JsonValue JsonValue::parse(const std::string& json_str) {
    JsonParser parser(json_str);
    return parser.parse();
}

//=============================================================================
// ConfigParser
//=============================================================================

ViewConfig ConfigParser::parse_json(const std::string& json_str) {
    ViewConfig config;
    config.init_defaults();
    
    JsonValue root = JsonValue::parse(json_str);
    if (!root.is_object()) {
        return config;
    }
    
    // Parse view_type
    if (root.has("view_type")) {
        config.view_type = parse_view_type(root["view_type"].as_string());
    }
    
    // Parse layer_range
    if (root.has("layer_range") && root["layer_range"].is_array()) {
        const auto& arr = root["layer_range"].as_array();
        if (arr.size() >= 1 && arr[0].is_number()) {
            config.layer_min = static_cast<size_t>(arr[0].as_int());
            config.has_layer_range = true;
        }
        if (arr.size() >= 2 && arr[1].is_number()) {
            config.layer_max = static_cast<size_t>(arr[1].as_int());
        }
    }
    
    // Parse visible_features
    if (root.has("visible_features") && root["visible_features"].is_object()) {
        const auto& obj = root["visible_features"].as_object();
        for (const auto& pair : obj) {
            if (pair.second.is_bool()) {
                libvgcode::EOptionType opt = parse_option_type(pair.first);
                if (opt != libvgcode::EOptionType::COUNT) {
                    config.option_visibility[opt] = pair.second.as_bool();
                }
            }
        }
    }
    
    // Parse extrusion_roles
    if (root.has("extrusion_roles") && root["extrusion_roles"].is_object()) {
        const auto& obj = root["extrusion_roles"].as_object();
        for (const auto& pair : obj) {
            libvgcode::EGCodeExtrusionRole role = parse_extrusion_role(pair.first);
            if (role != libvgcode::EGCodeExtrusionRole::COUNT) {
                if (pair.second.is_object()) {
                    if (pair.second.has("visible")) {
                        config.role_visibility[role] = pair.second["visible"].as_bool();
                    }
                    if (pair.second.has("color")) {
                        config.role_colors[role] = parse_color(pair.second["color"].as_string());
                    }
                }
            }
        }
    }
    
    // Parse tool_colors
    if (root.has("tool_colors") && root["tool_colors"].is_array()) {
        config.tool_colors.clear();
        for (const auto& item : root["tool_colors"].as_array()) {
            if (item.is_string()) {
                config.tool_colors.push_back(parse_color(item.as_string()));
            }
        }
    }
    
    // Parse output settings
    if (root.has("output") && root["output"].is_object()) {
        const auto& output = root["output"];
        if (output.has("width") && output["width"].is_number()) {
            config.width = output["width"].as_int();
        }
        if (output.has("height") && output["height"].is_number()) {
            config.height = output["height"].as_int();
        }
        if (output.has("background_color") && output["background_color"].is_string()) {
            config.background_color = parse_rgba(output["background_color"].as_string());
        }
    }
    
    return config;
}

ViewConfig ConfigParser::parse_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_json(buffer.str());
}

void ConfigParser::apply_config(libvgcode::Viewer& viewer, const ViewConfig& config) {
    // Set view type
    viewer.set_view_type(config.view_type);
    
    // Set layer range - always set it to ensure all layers are visible
    size_t layer_count = viewer.get_layers_count();
    if (layer_count > 0) {
        size_t min_layer = config.has_layer_range ? config.layer_min : 0;
        size_t max_layer = (config.has_layer_range && config.layer_max > 0) ? 
                           config.layer_max : (layer_count - 1);
        viewer.set_layers_view_range(min_layer, max_layer);
    }
    
    // Set the view visible range to show all vertices
    auto full_range = viewer.get_view_full_range();
    viewer.set_view_visible_range(full_range[0], full_range[1]);
    
    // Set option visibility
    for (const auto& pair : config.option_visibility) {
        if (viewer.is_option_visible(pair.first) != pair.second) {
            viewer.toggle_option_visibility(pair.first);
        }
    }
    
    // Set role visibility
    for (const auto& pair : config.role_visibility) {
        if (viewer.is_extrusion_role_visible(pair.first) != pair.second) {
            viewer.toggle_extrusion_role_visibility(pair.first);
        }
    }
    
    // Set role colors
    for (const auto& pair : config.role_colors) {
        viewer.set_extrusion_role_color(pair.first, pair.second);
    }
    
    // Set tool colors
    if (!config.tool_colors.empty()) {
        viewer.set_tool_colors(config.tool_colors);
    }
}

libvgcode::EViewType ConfigParser::parse_view_type(const std::string& type_str) {
    if (type_str == "FeatureType") return libvgcode::EViewType::FeatureType;
    if (type_str == "Height") return libvgcode::EViewType::Height;
    if (type_str == "Width") return libvgcode::EViewType::Width;
    if (type_str == "Speed") return libvgcode::EViewType::Speed;
    if (type_str == "ActualSpeed") return libvgcode::EViewType::ActualSpeed;
    if (type_str == "FanSpeed") return libvgcode::EViewType::FanSpeed;
    if (type_str == "Temperature") return libvgcode::EViewType::Temperature;
    if (type_str == "VolumetricFlowRate") return libvgcode::EViewType::VolumetricFlowRate;
    if (type_str == "ActualVolumetricFlowRate") return libvgcode::EViewType::ActualVolumetricFlowRate;
    if (type_str == "LayerTimeLinear") return libvgcode::EViewType::LayerTimeLinear;
    if (type_str == "LayerTimeLogarithmic") return libvgcode::EViewType::LayerTimeLogarithmic;
    if (type_str == "Tool") return libvgcode::EViewType::Tool;
    if (type_str == "ColorPrint") return libvgcode::EViewType::ColorPrint;
    return libvgcode::EViewType::FeatureType;
}

libvgcode::EOptionType ConfigParser::parse_option_type(const std::string& type_str) {
    std::string lower = type_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "travels") return libvgcode::EOptionType::Travels;
    if (lower == "wipes") return libvgcode::EOptionType::Wipes;
    if (lower == "retractions") return libvgcode::EOptionType::Retractions;
    if (lower == "unretractions") return libvgcode::EOptionType::Unretractions;
    if (lower == "seams") return libvgcode::EOptionType::Seams;
    if (lower == "tool_changes" || lower == "toolchanges") return libvgcode::EOptionType::ToolChanges;
    if (lower == "color_changes" || lower == "colorchanges") return libvgcode::EOptionType::ColorChanges;
    if (lower == "pause_prints" || lower == "pauseprints") return libvgcode::EOptionType::PausePrints;
    if (lower == "custom_gcodes" || lower == "customgcodes") return libvgcode::EOptionType::CustomGCodes;
    return libvgcode::EOptionType::COUNT;
}

libvgcode::EGCodeExtrusionRole ConfigParser::parse_extrusion_role(const std::string& role_str) {
    if (role_str == "Perimeter") return libvgcode::EGCodeExtrusionRole::Perimeter;
    if (role_str == "ExternalPerimeter") return libvgcode::EGCodeExtrusionRole::ExternalPerimeter;
    if (role_str == "OverhangPerimeter") return libvgcode::EGCodeExtrusionRole::OverhangPerimeter;
    if (role_str == "InternalInfill") return libvgcode::EGCodeExtrusionRole::InternalInfill;
    if (role_str == "SolidInfill") return libvgcode::EGCodeExtrusionRole::SolidInfill;
    if (role_str == "TopSolidInfill") return libvgcode::EGCodeExtrusionRole::TopSolidInfill;
    if (role_str == "Ironing") return libvgcode::EGCodeExtrusionRole::Ironing;
    if (role_str == "BridgeInfill") return libvgcode::EGCodeExtrusionRole::BridgeInfill;
    if (role_str == "GapFill") return libvgcode::EGCodeExtrusionRole::GapFill;
    if (role_str == "Skirt") return libvgcode::EGCodeExtrusionRole::Skirt;
    if (role_str == "SupportMaterial") return libvgcode::EGCodeExtrusionRole::SupportMaterial;
    if (role_str == "SupportMaterialInterface") return libvgcode::EGCodeExtrusionRole::SupportMaterialInterface;
    if (role_str == "WipeTower") return libvgcode::EGCodeExtrusionRole::WipeTower;
    if (role_str == "Custom") return libvgcode::EGCodeExtrusionRole::Custom;
    return libvgcode::EGCodeExtrusionRole::COUNT;
}

libvgcode::Color ConfigParser::parse_color(const std::string& color_str) {
    libvgcode::Color color = {128, 128, 128};
    
    if (color_str.empty()) return color;
    
    std::string hex = color_str;
    if (hex[0] == '#') {
        hex = hex.substr(1);
    }
    
    if (hex.size() >= 6) {
        try {
            color[0] = static_cast<uint8_t>(std::stoi(hex.substr(0, 2), nullptr, 16));
            color[1] = static_cast<uint8_t>(std::stoi(hex.substr(2, 2), nullptr, 16));
            color[2] = static_cast<uint8_t>(std::stoi(hex.substr(4, 2), nullptr, 16));
        } catch (...) {}
    }
    
    return color;
}

std::array<float, 4> ConfigParser::parse_rgba(const std::string& color_str) {
    libvgcode::Color rgb = parse_color(color_str);
    return {
        rgb[0] / 255.0f,
        rgb[1] / 255.0f,
        rgb[2] / 255.0f,
        1.0f
    };
}

} // namespace pygcode
