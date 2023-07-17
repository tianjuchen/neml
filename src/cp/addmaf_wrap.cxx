#include "pyhelp.h"

#include "cp/addmaf.h"

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)

namespace neml {

PYBIND11_MODULE(addmaf, m) {
  py::module::import("neml.objects");
  py::module::import("neml.cp.slipharden");
  py::module::import("neml.cp.sliprules");

  m.doc() = "Objects for the Dislocation density based AM 316SS model";

  py::class_<AMModel, HistoryNEMLObject,
      std::shared_ptr<AMModel>>(m,
                                                  "AMModel")
    .def(py::init([](py::args args, py::kwargs kwargs)
                  {
                    return create_object_python<AMModel>(
                        args, kwargs, {"mu", "kw1", "kw2", "ki1", "ki2"});
                  }))
    .def("varnames", &AMModel::varnames)
    .def("set_varnames", &AMModel::set_varnames)
    .def("populate_hist", &AMModel::populate_hist)
    .def("init_hist", &AMModel::init_hist)
    .def("wall_frac", &AMModel::wall_frac)
    .def("fmod", &AMModel::fmod)
    .def("macaulay", &AMModel::macaulay)
    .def("hist_to_tau", &AMModel::hist_to_tau)
    .def("dfdd", &AMModel::dfdd)
    .def("dfsigdd", &AMModel::dfsigdd)
    .def("d_hist_to_tau", &AMModel::d_hist_to_tau)
    .def("hist", &AMModel::hist)
    .def("d_hist_d_s", &AMModel::d_hist_d_s)
    .def("d_hist_d_h", &AMModel::d_hist_d_h)
    .def("d_hist_d_h_ext", &AMModel::d_hist_d_h_ext)
    ;

} // PYBIND!!_MODULE


}