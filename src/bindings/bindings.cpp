#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "snake/engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(snake_cpp, m) {
    m.doc() = "High-speed C++ Snake Engine for Reinforcement Learning";
    m.attr("TAIL_IS_SOLID") = SnakeEngine::TAIL_IS_SOLID;

    py::enum_<Action>(m, "Action")
        .value("FORWARD", Action::FORWARD)
        .value("LEFT", Action::LEFT)
        .value("RIGHT", Action::RIGHT)
        .export_values();

    py::enum_<Direction>(m, "Direction")
        .value("UP", Direction::UP)
        .value("RIGHT", Direction::RIGHT)
        .value("DOWN", Direction::DOWN)
        .value("LEFT", Direction::LEFT)
        .export_values();

    py::enum_<OrderEncoding>(m, "OrderEncoding")
        .value("RELATIVE", OrderEncoding::RELATIVE)
        .value("INVERSE", OrderEncoding::INVERSE)
        .export_values();

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("done", &StepResult::done)
        .def_readonly("truncated", &StepResult::truncated)
        .def_readonly("score", &StepResult::score)
        .def_readonly("reward", &StepResult::reward);

    py::class_<SnakeEngine>(m, "SnakeEngine")
        .def(py::init<int, int, float>(),
             py::arg("width") = 20, py::arg("height") = 20, py::arg("gamma") = 0.99f)
        .def("reset", &SnakeEngine::reset)
        .def("clone", [](const SnakeEngine& self) { return SnakeEngine(self); })
        .def("restore", [](SnakeEngine& self, const SnakeEngine& other) { self = other; })
        .def("reset_with_length", &SnakeEngine::resetWithLength, py::arg("length"))
        .def("step", &SnakeEngine::step)
        .def("get_width", &SnakeEngine::getWidth)
        .def("get_direction", [](const SnakeEngine& self) { return static_cast<int>(self.getSnakeDirection()); })
        .def("get_height", &SnakeEngine::getHeight)
        .def_property("gamma", &SnakeEngine::getGamma, &SnakeEngine::setGamma)
        .def_property("order_encoding", &SnakeEngine::getOrderEncoding, &SnakeEngine::setOrderEncoding)
        .def("get_danger", [](const SnakeEngine& self) {
            const auto d = self.getDanger();
            return py::make_tuple(d[0], d[1], d[2]);
        })
        .def("get_tail_segments", [](const SnakeEngine& self, int count) {
            const auto segments = self.getTailSegments(count);
            py::array_t<int32_t> out({static_cast<py::ssize_t>(segments.size()), static_cast<py::ssize_t>(2)});
            auto r = out.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(segments.size()); ++i) {
                r(i, 0) = segments[static_cast<size_t>(i)].x;
                r(i, 1) = segments[static_cast<size_t>(i)].y;
            }
            return out;
        }, py::arg("count"))
        .def("get_body", [](const SnakeEngine& self) {
            py::list cells;
            for (const auto& segment : self.getSnakeBody()) {
                cells.append(py::make_tuple(segment.x, segment.y));
            }
            return cells;
        })
        .def_readonly_static("num_channels", &SnakeEngine::NUM_CHANNELS)
        .def("get_score", [](const SnakeEngine& self) {
            return static_cast<int>(self.getSnakeBody().size()) - 2;
        })
        .def("get_board", [](py::object obj) {
            const SnakeEngine& self = obj.cast<const SnakeEngine&>();
            const auto& buf = self.getBoardBuffer();
            const py::ssize_t c = SnakeEngine::NUM_CHANNELS;
            const py::ssize_t h = self.getHeight();
            const py::ssize_t w = self.getWidth();
            return py::array_t<uint8_t>(
                {c, h, w},
                {h * w, w, static_cast<py::ssize_t>(1)},
                buf.data(),
                obj
            );
        });
}