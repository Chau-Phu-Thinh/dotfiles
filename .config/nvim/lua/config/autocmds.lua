-- Autocmds are automatically loaded on the VeryLazy event
-- Default autocmds that are always set: https://github.com/LazyVim/LazyVim/blob/main/lua/lazyvim/config/autocmds.lua
--
-- Add any additional autocmds here
-- with `vim.api.nvim_create_autocmd`
--
-- Or remove existing autocmds by their group name (which is prefixed with `lazyvim_` for the defaults)
-- e.g. vim.api.nvim_del_augroup_by_name("lazyvim_wrap_spell")

vim.api.nvim_create_user_command("WatchRun", function()
  local overseer = require("overseer")
  overseer.run_task({ name = "run script", autostart = false }, function(task)
    if task then
      task:add_component({ "restart_on_save", paths = { vim.fn.expand("%:p") } })
      task:start()
      task:open_output("vertical")
    else
      vim.notify("WatchRun not supported for filetype " .. vim.bo.filetype, vim.log.levels.ERROR)
    end
  end)
end, {})
