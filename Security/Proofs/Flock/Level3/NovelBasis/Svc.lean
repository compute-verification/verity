/-
`NovelBasis`'s table `svC` of `s_i(β_i)`, apart from the checks of its entries so they build in parallel.
-/
import Flock.Ligerito

open Flock

namespace FlockLevel3

/-- `s_i(β_i)` for `i < 40`, computed natively; each is checked by the kernel on its own (`svC_step_i`). -/
def svC : Array F128 := #[⟨0x1, 0x0⟩,
  ⟨0x6, 0x0⟩,
  ⟨0x168, 0x0⟩,
  ⟨0x1166880, 0x0⟩,
  ⟨0x116166868808000, 0x1⟩,
  ⟨0x6880808780878c62, 0x116166816686880⟩,
  ⟨0x977dcfdeca5763a3, 0xe4e99706977dcfa5⟩,
  ⟨0xb9db1ae235f97b78, 0xd738e365b46abbf3⟩,
  ⟨0x30f75909af71f18, 0xdb3f8356fe4c8aae⟩,
  ⟨0x7206f70a338f4695, 0xd8e6a10ef60e4734⟩,
  ⟨0x51dd7adac28a6297, 0x8ee9519cd23466ce⟩,
  ⟨0x5ff07b8f5c1f9d6a, 0xfbd61fe266c82bf2⟩,
  ⟨0x88c3d85b6c32f957, 0x8f4a1c96f72deb13⟩,
  ⟨0xa79d9858273d54c, 0x83ad3f8aaaa6e6a2⟩,
  ⟨0x48d8ee13015f698e, 0x69a798e77cf5b06e⟩,
  ⟨0xe06ba4468e8c21cc, 0x8ee54c8e8ae297ba⟩,
  ⟨0xaaa2b91320af4f9a, 0x267da383a0b4a6ec⟩,
  ⟨0x2d24309579f502ec, 0x30f556da177d94c8⟩,
  ⟨0x517cd50cfd167ed4, 0x4a0793046f3e66c0⟩,
  ⟨0xfb14b88ada747186, 0x1399f4f9afb56ae0⟩,
  ⟨0x781b54b7ed1db6cb, 0x836a410cea8384d2⟩,
  ⟨0x7458fdfb343d809f, 0xb1c201bdfda6af9d⟩,
  ⟨0xcde3753a43aa9865, 0x6379afc27b1cb40⟩,
  ⟨0xd119d7ce1e4de8bd, 0xf8bd0542e9d1c09b⟩,
  ⟨0x8447841ac1efb10c, 0xe19b1423499f5bdd⟩,
  ⟨0xd88934c2e6fe7c14, 0x99baf0a32c6f9cb9⟩,
  ⟨0xdcb4f9b26cd865b8, 0xc8a3ee15787627e0⟩,
  ⟨0xb64bd5f8acb8dbc1, 0xde2a33f838331e7c⟩,
  ⟨0xeb7a1fbbf20aaa6f, 0x916f7d658430ba75⟩,
  ⟨0x54e2cc4459485d3d, 0x79d06d5f044e0e1f⟩,
  ⟨0x25f1ddcbd1607b0d, 0x12265f9b8d8868b2⟩,
  ⟨0x57169ceff2335d55, 0xb3961d2b903d11cd⟩,
  ⟨0x51821cfd88fe99a0, 0x3c625c142e1d515d⟩,
  ⟨0x329b9e4fe6f2c912, 0xbb7481dbb408e234⟩,
  ⟨0xafd85dd470c6d22c, 0x990f71d382817816⟩,
  ⟨0x997fe478c0b53f76, 0x26c36b2858dde4cc⟩,
  ⟨0xe9b842f22c8900a4, 0x807b0eb99eb7b556⟩,
  ⟨0xbb97a29817f15f01, 0x12426ef6bda5a89a⟩,
  ⟨0x64dfd5d9402b863, 0x67828ae497510edd⟩,
  ⟨0x3d37f769aee1d2cf, 0xef043cb9fc49b6d0⟩]

end FlockLevel3
