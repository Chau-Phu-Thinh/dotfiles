-- bootstrap lazy.nvim, LazyVim and your plugins
vim.cmd("set background&")

require("config.lazy")

require("overseer").setup({})

vim.opt.clipboard = "unnamedplus"
