require("conform").setup({
  formatters_by_ft = {
    json = { "prettier" },
    jsonc = { "prettier" },
  },

  format_on_save = {
    lsp_fallback = "fallback",
    timeout_ms = 500, -- Thời gian chờ format tối đa (nửa giây)
  },
})
