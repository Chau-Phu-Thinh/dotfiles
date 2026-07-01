return {
  "stevearc/conform.nvim",
  event = "VeryLazy", -- alternative if you prefer

  opts = {
    formatters_by_ft = {
      json = { "prettier" },
      jsonc = { "prettier" },
      markdown = { "prettier" },
      cpp = { "clang_format" },
      c = { "clang_format" },
    },
  },
}
