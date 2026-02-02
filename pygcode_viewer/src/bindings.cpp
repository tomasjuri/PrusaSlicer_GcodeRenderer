///
/// pygcode_viewer - Python Bindings
/// pybind11 module exposing the GCode viewer API
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gcode_viewer.hpp"
#include "camera.hpp"
#include "config_parser.hpp"
#include "bed_renderer.hpp"

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
    
    // BedConfig struct
    py::class_<pygcode::BedConfig>(m, "BedConfig")
        .def(py::init<>())
        .def_readwrite("size_x", &pygcode::BedConfig::size_x)
        .def_readwrite("size_y", &pygcode::BedConfig::size_y)
        .def_readwrite("origin_x", &pygcode::BedConfig::origin_x)
        .def_readwrite("origin_y", &pygcode::BedConfig::origin_y)
        .def_readwrite("outline_color", &pygcode::BedConfig::outline_color)
        .def_readwrite("grid_color", &pygcode::BedConfig::grid_color)
        .def_readwrite("show_outline", &pygcode::BedConfig::show_outline)
        .def_readwrite("show_grid", &pygcode::BedConfig::show_grid)
        .def_readwrite("grid_spacing", &pygcode::BedConfig::grid_spacing)
        .def_readwrite("line_width", &pygcode::BedConfig::line_width)
        .def_readwrite("z_position", &pygcode::BedConfig::z_position);
    
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
        
        .def("set_bed_config", &pygcode::GCodeViewerCore::set_bed_config,
             py::arg("config"),
             "Set bed configuration")
        
        .def("get_bed_config", &pygcode::GCodeViewerCore::get_bed_config,
             py::return_value_policy::reference,
             "Get bed configuration")
        
        .def("render_to_file",
             [](pygcode::GCodeViewerCore& self,
                const std::string& output_path,
                int width,
                int height,
                const py::dict& camera_dict,
                const py::object& config) {
                 
                 // Check for extrinsic camera parameters (rvec/tvec or rotation/translation)
                 // Priority: rvec/tvec > rotation/translation > position/target/up
                 bool use_extrinsics = false;
                 float rvec_x = 0, rvec_y = 0, rvec_z = 0;
                 float tvec_x = 0, tvec_y = 0, tvec_z = 0;
                 pygcode::Mat3x3 rotation_matrix;
                 bool has_rotation_matrix = false;
                 
                 // Check for rvec/tvec (Rodrigues format from cv2.solvePnP)
                 if (camera_dict.contains("rvec") && camera_dict.contains("tvec")) {
                     py::sequence rvec = camera_dict["rvec"].cast<py::sequence>();
                     py::sequence tvec = camera_dict["tvec"].cast<py::sequence>();
                     if (rvec.size() >= 3 && tvec.size() >= 3) {
                         rvec_x = rvec[0].cast<float>();
                         rvec_y = rvec[1].cast<float>();
                         rvec_z = rvec[2].cast<float>();
                         tvec_x = tvec[0].cast<float>();
                         tvec_y = tvec[1].cast<float>();
                         tvec_z = tvec[2].cast<float>();
                         use_extrinsics = true;
                     }
                 }
                 // Check for rotation/translation (3x3 matrix format)
                 else if (camera_dict.contains("rotation") && camera_dict.contains("translation")) {
                     py::sequence rot = camera_dict["rotation"].cast<py::sequence>();
                     py::sequence trans = camera_dict["translation"].cast<py::sequence>();
                     if (rot.size() >= 3 && trans.size() >= 3) {
                         // Parse 3x3 rotation matrix
                         for (int i = 0; i < 3; ++i) {
                             py::sequence row = rot[i].cast<py::sequence>();
                             if (row.size() >= 3) {
                                 rotation_matrix[i][0] = row[0].cast<float>();
                                 rotation_matrix[i][1] = row[1].cast<float>();
                                 rotation_matrix[i][2] = row[2].cast<float>();
                             }
                         }
                         tvec_x = trans[0].cast<float>();
                         tvec_y = trans[1].cast<float>();
                         tvec_z = trans[2].cast<float>();
                         has_rotation_matrix = true;
                         use_extrinsics = true;
                     }
                 }
                 
                 // Fall back to position/target/up
                 float pos_x = 100, pos_y = 100, pos_z = 100;
                 float target_x = 0, target_y = 0, target_z = 50;
                 float up_x = 0, up_y = 0, up_z = 1;  // Default Z-up
                 
                 if (!use_extrinsics) {
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
                     
                     // Extract up vector (important for camera orientation!)
                     if (camera_dict.contains("up")) {
                         py::sequence up = camera_dict["up"].cast<py::sequence>();
                         if (up.size() >= 3) {
                             up_x = up[0].cast<float>();
                             up_y = up[1].cast<float>();
                             up_z = up[2].cast<float>();
                         }
                     }
                 }
                 
                 // Extract optional near/far planes
                 float near_plane = 0.1f;
                 float far_plane = 10000.0f;
                 if (camera_dict.contains("near")) {
                     near_plane = camera_dict["near"].cast<float>();
                 }
                 if (camera_dict.contains("far")) {
                     far_plane = camera_dict["far"].cast<float>();
                 }
                 
                 // Extract optional camera intrinsics
                 pygcode::CameraIntrinsics intrinsics;
                 if (camera_dict.contains("intrinsics")) {
                     py::dict intr = camera_dict["intrinsics"].cast<py::dict>();
                     
                     // Check for camera_matrix (3x3 array format from OpenCV calibration)
                     if (intr.contains("camera_matrix")) {
                         py::sequence matrix = intr["camera_matrix"].cast<py::sequence>();
                         if (matrix.size() >= 3) {
                             py::sequence row0 = matrix[0].cast<py::sequence>();
                             py::sequence row1 = matrix[1].cast<py::sequence>();
                             
                             intrinsics.fx = row0[0].cast<float>();
                             intrinsics.cx = row0[2].cast<float>();
                             intrinsics.fy = row1[1].cast<float>();
                             intrinsics.cy = row1[2].cast<float>();
                             intrinsics.is_valid = true;
                         }
                     }
                     // Or individual fx, fy, cx, cy values
                     else {
                         if (intr.contains("fx")) intrinsics.fx = intr["fx"].cast<float>();
                         if (intr.contains("fy")) intrinsics.fy = intr["fy"].cast<float>();
                         if (intr.contains("cx")) intrinsics.cx = intr["cx"].cast<float>();
                         if (intr.contains("cy")) intrinsics.cy = intr["cy"].cast<float>();
                         if (intrinsics.fx > 0 && intrinsics.fy > 0) {
                             intrinsics.is_valid = true;
                         }
                     }
                     
                     // Image size from intrinsics
                     if (intr.contains("image_size")) {
                         py::sequence size = intr["image_size"].cast<py::sequence>();
                         if (size.size() >= 2) {
                             intrinsics.image_width = size[0].cast<int>();
                             intrinsics.image_height = size[1].cast<int>();
                         }
                     } else {
                         // Default to render size if not specified
                         intrinsics.image_width = width;
                         intrinsics.image_height = height;
                     }
                 }
                 
                 // Convert config to JSON string
                 std::string config_json;
                 if (py::isinstance<py::dict>(config)) {
                     config_json = dict_to_json(config.cast<py::dict>());
                 } else if (py::isinstance<py::str>(config)) {
                     config_json = config.cast<std::string>();
                 }
                 
                 // Call appropriate render method based on camera parameters
                 if (use_extrinsics) {
                     if (has_rotation_matrix) {
                         self.render_to_file_extrinsics(output_path, width, height,
                                                        rotation_matrix,
                                                        tvec_x, tvec_y, tvec_z,
                                                        near_plane, far_plane,
                                                        intrinsics,
                                                        config_json);
                     } else {
                         self.render_to_file_rodrigues(output_path, width, height,
                                                       rvec_x, rvec_y, rvec_z,
                                                       tvec_x, tvec_y, tvec_z,
                                                       near_plane, far_plane,
                                                       intrinsics,
                                                       config_json);
                     }
                 } else {
                     self.render_to_file(output_path, width, height,
                                         pos_x, pos_y, pos_z,
                                         target_x, target_y, target_z,
                                         up_x, up_y, up_z,
                                         near_plane, far_plane,
                                         intrinsics,
                                         config_json);
                 }
             },
             py::arg("output_path"),
             py::arg("width"),
             py::arg("height"),
             py::arg("camera"),
             py::arg("config"),
             R"doc(Render GCODE to a PNG file.

Camera dict supports multiple formats:
- Position/target/up: {"position": [x,y,z], "target": [x,y,z], "up": [x,y,z]}
- Rodrigues (from solvePnP): {"rvec": [rx,ry,rz], "tvec": [tx,ty,tz]}
- Rotation matrix: {"rotation": [[r00,r01,r02],...], "translation": [tx,ty,tz]}

Optional camera dict keys:
- "near": near clipping plane (default 0.1)
- "far": far clipping plane (default 10000)
- "intrinsics": camera intrinsics dict with camera_matrix or fx/fy/cx/cy values
)doc")
        ;
    
    // Version info
    m.attr("__version__") = "0.1.0";
}
