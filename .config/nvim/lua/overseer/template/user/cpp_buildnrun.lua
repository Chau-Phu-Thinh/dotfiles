return {
  name = "C/C++ Build & Run",
  builder = function()
    local file = vim.fn.expand("%:p")
    local exe = vim.fn.expand("%:t:r")

    local compiler = (vim.bo.filetype == "c") and "gcc" or "g++"
    local std = (vim.bo.filetype == "c") and "" or "-std=c++20"

    return {
      cmd = compiler,
      args = vim.tbl_filter(function(x)
        return x ~= ""
      end, {
        file,
        "-o",
        exe,
        "-Wall",
        "-g",
        std,
      }),
      components = {
        { "on_output_quickfix", set_diagnostics = true },
        "on_result_diagnostics",
        "default",
      },
    }
  end,
  condition = {
    filetype = { "c", "cpp", "c++" },
  },
}
