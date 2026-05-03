-- Keymaps are automatically loaded on the VeryLazy event
-- Default keymaps that are always set: https://github.com/LazyVim/LazyVim/blob/main/lua/lazyvim/config/keymaps.lua
-- Add any additional keymaps here
local map = vim.keymap.set
local wk = require("which-key")
map("n", "<leader>bD", function()
  Snacks.bufdelete()
end, { desc = "Delete Buffer" })
map("n", "<leader>bd", "<cmd>:bd<cr>", { desc = "Delete Buffer and Window" })

wk.add({
  { "<leader>o", group = "Overseer" },
  { "<leader>ot", "<cmd>OverseerTaskAction<cr>", desc = "TaskAction", mode = "n" },
  { "<leader>os", "<cmd>OverseerShell<cr>", desc = "Shell", mode = "n" },
  { "<leader>oc", "<cmd>OverseerClose<cr>", desc = "Close", mode = "n" },
  { "<leader>oo", "<cmd>OverseerOpen<cr>", desc = "Open", mode = "n" },
  { "<leader>or", group = "Run or Restart" },
  { "<leader>ort", "<cmd>OverseerRestart<cr>", desc = "Restart", mode = "n" },
  { "<leader>orr", "<cmd>OverseerRun<cr>", desc = "Run", mode = "n" },
  { "<leader>ow", "<cmd>WatchRun<cr>", desc = "WatchRun", mode = "n" },
})
