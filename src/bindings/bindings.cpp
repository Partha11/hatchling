#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "snake/engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(snake_cpp, m) {
    m.doc() = "High-speed C++ Snake Engine for Reinforcement Learning";

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

    py::class_<StepResult>(m, "StepResult")
        .def_readonly("done", &StepResult::done)
        .def_readonly("truncated", &StepResult::truncated)
        .def_readonly("score", &StepResult::score)
        .def_readonly("reward", &StepResult::reward);

    py::class_<SnakeEngine>(m, "SnakeEngine")
        .def(py::init<int, int, float>(),
             py::arg("width") = 20, py::arg("height") = 20, py::arg("gamma") = 0.99f)
        .def("reset", &SnakeEngine::reset)
        .def("step", &SnakeEngine::step)
        .def("get_width", &SnakeEngine::getWidth)
        .def("get_height", &SnakeEngine::getHeight)
        .def_property("gamma", &SnakeEngine::getGamma, &SnakeEngine::setGamma)
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