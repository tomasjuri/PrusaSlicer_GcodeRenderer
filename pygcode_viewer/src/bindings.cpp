///
/// pygcode_viewer - Python Bindings
/// pybind11 module exposing the GCode viewer API
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gcode_viewer.hpp"
#include "camera.hpp"
#include "config_parser.hpp"

#include <sstream>

namespace py = pybind11;

// Helper to convert Python dict to JSON string
std::string dict_to_json(const py::dict& d);

std::string value_to_json(const py::handle& obj) {
    if (py::isinstance<py::none>(obj)) {
        return "null";
    }
    if (py::isinstance<py::bool_>(obj)) {
        return obj.cast<bool>() ? "true" : "false";
    }
    if (py::isinstance<py::int_>(obj)) {
        return std::to_string(obj.cast<long long>());
    }
    if (py::isinstance<py::float_>(obj)) {
        std::ostringstream oss;
        oss << obj.cast<double>();
        return oss.str();
    }
    if (py::isinstance<py::str>(obj)) {
        std::string s = obj.cast<std::string>();
        // Escape string for JSON
        std::string result = "\"";
        for (char c : s) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c;
            }
        }
        result += "\"";
        return result;
    }
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = obj.cast<py::sequence>();
        std::string result = "[";
        bool first = true;
        for (size_t i = 0; i < seq.size(); ++i) {
            if (!first) result += ",";
            first = false;
            result += value_to_json(seq[i]);
        }
        result += "]";
        return result;
    }
    if (py::isinstance<py::dict>(obj)) {
        return dict_to_json(obj.cast<py::dict>());
    }
    return "null";
}

std::string dict_to_json(const py::dict& d) {
    std::string result = "{";
    bool first = true;
    for (auto item : d) {
        if (!first) result += ",";
        first = false;
        result += "\"" + item.first.cast<std::string>() + "\":";
        result += value_to_json(item.second);
    }
    result += "}";
    return result;
}

PYBIND11_MODULE(_pygcode_viewer, m) {
    m.doc() = "Python bindings for PrusaSlicer GCode visualization";
    
    // GCodeViewerCore class
    py::class_<pygcode::GCodeViewerCore>(m, "GCodeViewerCore")
        .def(py::init<>())
        
        .def("load", &pygcode::GCodeViewerCore::load,
             py::arg("filepath"),
             "Load a GCODE file")
        
        .def("is_loaded", &pygcode::GCodeViewerCore::is_loaded,
             "Check if a GCODE file is loaded")
        
        .def("get_bounding_box", &pygcode::GCodeViewerCore::get_bounding_box,
             "Get bounding box as (min_x, min_y, min_z, max_x, max_y, max_z)")
        
        .def("get_layer_count", &pygcode::GCodeViewerCore::get_layer_count,
             "Get the number of layers")
        
        .def("get_estimated_time", &pygcode::GCodeViewerCore::get_estimated_time,
             "Get estimated print time in seconds")
        
        .def("set_layer_range", &pygcode::GCodeViewerCore::set_layer_range,
             py::arg("min_layer"), py::arg("max_layer"),
             "Set the layer range to display")
        
        .def("set_view_type", &pygcode::GCodeViewerCore::set_view_type,
             py::arg("view_type"),
             "Set view type (FeatureType, Height, Width, Speed, etc.)")
        
        .def("apply_config", &pygcode::GCodeViewerCore::apply_config,
             py::arg("config_json"),
             "Apply JSON configuration")
        
        .def("render_to_file",
             [](pygcode::GCodeViewerCore& self,
                const std::string& output_path,
                int width,
                int height,
                const py::dict& camera_dict,
                const py::object& config) {
                 
                 // Extract camera parameters
                 float pos_x = 100, pos_y = 100, pos_z = 100;
                 float target_x = 0, target_y = 0, target_z = 50;
                 
                 if (camera_dict.contains("position")) {
                     py::sequence pos = camera_dict["position"].cast<py::sequence>();
                     if (pos.size() >= 3) {
                         pos_x = pos[0].cast<float>();
                         pos_y = pos[1].cast<float>();
                         pos_z = pos[2].cast<float>();
                     }
                 }
                 
                 if (camera_dict.contains("target")) {
                     py::sequence tgt = camera_dict["target"].cast<py::sequence>();
                     if (tgt.size() >= 3) {
                         target_x = tgt[0].cast<float>();
                         target_y = tgt[1].cast<float>();
                         target_z = tgt[2].cast<float>();
                     }
                 }
                 
                 // Convert config to JSON string
                 std::string config_json;
                 if (py::isinstance<py::dict>(config)) {
                     config_json = dict_to_json(config.cast<py::dict>());
                 } else if (py::isinstance<py::str>(config)) {
                     config_json = config.cast<std::string>();
                 }
                 
                 self.render_to_file(output_path, width, height,
                                     pos_x, pos_y, pos_z,
                                     target_x, target_y, target_z,
                                     config_json);
             },
             py::arg("output_path"),
             py::arg("width"),
             py::arg("height"),
             py::arg("camera"),
             py::arg("config"),
             "Render GCODE to a PNG file")
        ;
    
    // Version info
    m.attr("__version__") = "0.1.0";
}
