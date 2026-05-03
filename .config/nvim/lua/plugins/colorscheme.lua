return {
  "uhs-robert/oasis.nvim",
  lazy = false,
  priority = 1000,

  config = function()
    local is_light = vim.o.background == "light"

    require("oasis").setup({
      transparent = true,
      style = is_light and "starlight" or "abyss",

      dark_intensity = 5,
      light_intensity = 1,
      match_paren_bg = true,
    })

    vim.cmd.colorscheme("oasis")
  end,
}
