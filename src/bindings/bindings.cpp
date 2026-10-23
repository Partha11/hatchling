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
        .def_readonly("score", &StepResult::score)
        .def_readonly("reward", &StepResult::reward);

    py::class_<SnakeEngine>(m, "SnakeEngine")
        .def(py::init<int, int>(), py::arg("width") = 20, py::arg("height") = 20)
        .def("reset", &SnakeEngine::reset)
        .def("step", &SnakeEngine::step)
        .def("get_width", &SnakeEngine::getWidth)
        .def("get_height", &SnakeEngine::getHeight)
        .def("get_score", [](const SnakeEngine& self) {
            return static_cast<int>(self.getSnakeBody().size()) - 2;
        })
        .def("get_board", [](SnakeEngine& self) {
            const auto& buf = self.getBoardBuffer();
            return py::array_t<uint8_t>(
                {4, self.getHeight(), self.getWidth()},
                {static_cast<ssize_t>(self.getHeight() * self.getWidth()),
                 static_cast<ssize_t>(self.getWidth()),
                 static_cast<ssize_t>(1)},
                buf.data(),
                py::cast(self)
            );
        });
}