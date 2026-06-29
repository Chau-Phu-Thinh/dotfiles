return {
  {
    "uhs-robert/oasis.nvim",
    lazy = false,
    priority = 1000,
    config = function()
      require("oasis").setup({
        style = "starlight",
        light_style = "abyss",
        light_intensity = 1,
      })
    end,
  },

  {
    "LazyVim/LazyVim",
    opts = {
      colorscheme = "oasis",
    },
  },
}
