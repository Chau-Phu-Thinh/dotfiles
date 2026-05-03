-- ~/.config/nvim/lua/overseer/template/user/run_script.lua
return {
  name = "run script",
  builder = function()
    local file = vim.fn.expand("%:p") -- full path to source file
    local exe = vim.fn.expand("%:t:r") -- filename without extension (e.g. "main")
    local ft = vim.bo.filetype
    local dir = vim.fn.expand("%:p:h")

    local cmd = {}
    local args = {}

    if ft == "go" then
      cmd = { "go", "run", file }
    elseif ft == "python" then
      cmd = { "python3", file }
    elseif ft == "c" then
      cmd = {
        "bash",
        "-c",
        string.format("gcc %s -o %s/%s -Wall -g && %s/%s", file, dir, exe, dir, exe),
      }
    elseif ft == "cpp" or ft == "c++" then
      cmd = {
        "bash",
        "-c",
        string.format("g++ %s -o %s -Wall -g -std=c++20 && ./%s", vim.fn.shellescape(file), exe, exe),
      }
    else
      -- For scripts (sh, bash, etc.)
      cmd = { file }
    end

    return {
      cmd = cmd,
      components = {
        { "on_output_quickfix", set_diagnostics = true },
        "on_result_diagnostics",
        "default",
      },
    }
  end,

  condition = {
    filetype = { "sh", "python", "go", "c", "cpp", "c++" },
  },
}
