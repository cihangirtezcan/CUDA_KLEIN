#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ctime>
#define BLOCKS				1024
#define THREADS				1024
#define TRIALS				1024*1024

// __byte_perm Constants
#define SHIFT_1_RIGHT			17185  // 0x00004321U i.e. ( >> 8 )
#define SHIFT_2_RIGHT			21554  // 0x00005432U i.e. ( >> 16 )
#define SHIFT_3_RIGHT			25923  // 0x00006543U i.e. ( >> 24 )

__int64 trial = 1, keys = 10;
double PCFreq = 0.0;
__int64 CounterStart = 0;
#define bit8 unsigned char
#define bit32 unsigned int
#define bit64 unsigned __int64 

bit32 T0[256] = { 0xee777799, 0xe874749c, 0xf47a7a8e, 0xf279798b, 0xe2717193, 0xfe7f7f81, 0xf67b7b8d, 0xe0707090, 0xf87c7c84, 0xe6737395, 0xe4727296, 0xec76769a, 0xf0787888, 0xfc7e7e82, 0xfa7d7d87, 0xea75759f, 0x8e4747c9, 0x884444cc, 0x944a4ade, 0x924949db, 0x824141c3, 0x9e4f4fd1, 0x964b4bdd, 0x804040c0, 0x984c4cd4, 0x864343c5, 0x844242c6, 0x8c4646ca, 0x904848d8, 0x9c4e4ed2, 0x9a4d4dd7, 0x8a4545cf, 0x55a7a7f2, 0x53a4a4f7, 0x4faaaae5, 0x49a9a9e0, 0x59a1a1f8, 0x45afafea, 0x4dababe6, 0x5ba0a0fb, 0x43acacef, 0x5da3a3fe, 0x5fa2a2fd, 0x57a6a6f1, 0x4ba8a8e3, 0x47aeaee9, 0x41adadec, 0x51a5a5f4, 0x359797a2, 0x339494a7, 0x2f9a9ab5, 0x299999b0, 0x399191a8, 0x259f9fba, 0x2d9b9bb6, 0x3b9090ab, 0x239c9cbf, 0x3d9393ae, 0x3f9292ad, 0x379696a1, 0x2b9898b3, 0x279e9eb9, 0x219d9dbc, 0x319595a4, 0x2e171739, 0x2814143c, 0x341a1a2e, 0x3219192b, 0x22111133, 0x3e1f1f21, 0x361b1b2d, 0x20101030, 0x381c1c24, 0x26131335, 0x24121236, 0x2c16163a, 0x30181828, 0x3c1e1e22, 0x3a1d1d27, 0x2a15153f, 0xf5f7f702, 0xf3f4f407, 0xeffafa15, 0xe9f9f910, 0xf9f1f108, 0xe5ffff1a, 0xedfbfb16, 0xfbf0f00b, 0xe3fcfc1f, 0xfdf3f30e, 0xfff2f20d, 0xf7f6f601, 0xebf8f813, 0xe7fefe19, 0xe1fdfd1c, 0xf1f5f504, 0x75b7b7c2, 0x73b4b4c7, 0x6fbabad5, 0x69b9b9d0, 0x79b1b1c8, 0x65bfbfda, 0x6dbbbbd6, 0x7bb0b0cb, 0x63bcbcdf, 0x7db3b3ce, 0x7fb2b2cd, 0x77b6b6c1, 0x6bb8b8d3, 0x67bebed9, 0x61bdbddc, 0x71b5b5c4, 0x0e070709, 0x0804040c, 0x140a0a1e, 0x1209091b, 0x02010103, 0x1e0f0f11, 0x160b0b1d, 0x00000000, 0x180c0c14, 0x06030305, 0x04020206, 0x0c06060a, 0x10080818, 0x1c0e0e12, 0x1a0d0d17, 0x0a05050f, 0x95c7c752, 0x93c4c457, 0x8fcaca45, 0x89c9c940, 0x99c1c158, 0x85cfcf4a, 0x8dcbcb46, 0x9bc0c05b, 0x83cccc4f, 0x9dc3c35e, 0x9fc2c25d, 0x97c6c651, 0x8bc8c843, 0x87cece49, 0x81cdcd4c, 0x91c5c554, 0x6e373759, 0x6834345c, 0x743a3a4e, 0x7239394b, 0x62313153, 0x7e3f3f41, 0x763b3b4d, 0x60303050, 0x783c3c44, 0x66333355, 0x64323256, 0x6c36365a, 0x70383848, 0x7c3e3e42, 0x7a3d3d47, 0x6a35355f, 0x4e272769, 0x4824246c, 0x542a2a7e, 0x5229297b, 0x42212163, 0x5e2f2f71, 0x562b2b7d, 0x40202060, 0x582c2c74, 0x46232365, 0x44222266, 0x4c26266a, 0x50282878, 0x5c2e2e72, 0x5a2d2d77, 0x4a25256f, 0xce6767a9, 0xc86464ac, 0xd46a6abe, 0xd26969bb, 0xc26161a3, 0xde6f6fb1, 0xd66b6bbd, 0xc06060a0, 0xd86c6cb4, 0xc66363a5, 0xc46262a6, 0xcc6666aa, 0xd06868b8, 0xdc6e6eb2, 0xda6d6db7, 0xca6565af, 0x15878792, 0x13848497, 0x0f8a8a85, 0x09898980, 0x19818198, 0x058f8f8a, 0x0d8b8b86, 0x1b80809b, 0x038c8c8f, 0x1d83839e, 0x1f82829d, 0x17868691, 0x0b888883, 0x078e8e89, 0x018d8d8c, 0x11858594, 0xd5e7e732, 0xd3e4e437, 0xcfeaea25, 0xc9e9e920, 0xd9e1e138, 0xc5efef2a, 0xcdebeb26, 0xdbe0e03b, 0xc3ecec2f, 0xdde3e33e, 0xdfe2e23d, 0xd7e6e631, 0xcbe8e823, 0xc7eeee29, 0xc1eded2c, 0xd1e5e534, 0xb5d7d762, 0xb3d4d467, 0xafdada75, 0xa9d9d970, 0xb9d1d168, 0xa5dfdf7a, 0xaddbdb76, 0xbbd0d06b, 0xa3dcdc7f, 0xbdd3d36e, 0xbfd2d26d, 0xb7d6d661, 0xabd8d873, 0xa7dede79, 0xa1dddd7c, 0xb1d5d564, 0xae5757f9, 0xa85454fc, 0xb45a5aee, 0xb25959eb, 0xa25151f3, 0xbe5f5fe1, 0xb65b5bed, 0xa05050f0, 0xb85c5ce4, 0xa65353f5, 0xa45252f6, 0xac5656fa, 0xb05858e8, 0xbc5e5ee2, 0xba5d5de7, 0xaa5555ff};
bit32 T1[256] = { 0x99ee7777, 0x9ce87474, 0x8ef47a7a, 0x8bf27979, 0x93e27171, 0x81fe7f7f, 0x8df67b7b, 0x90e07070, 0x84f87c7c, 0x95e67373, 0x96e47272, 0x9aec7676, 0x88f07878, 0x82fc7e7e, 0x87fa7d7d, 0x9fea7575, 0xc98e4747, 0xcc884444, 0xde944a4a, 0xdb924949, 0xc3824141, 0xd19e4f4f, 0xdd964b4b, 0xc0804040, 0xd4984c4c, 0xc5864343, 0xc6844242, 0xca8c4646, 0xd8904848, 0xd29c4e4e, 0xd79a4d4d, 0xcf8a4545, 0xf255a7a7, 0xf753a4a4, 0xe54faaaa, 0xe049a9a9, 0xf859a1a1, 0xea45afaf, 0xe64dabab, 0xfb5ba0a0, 0xef43acac, 0xfe5da3a3, 0xfd5fa2a2, 0xf157a6a6, 0xe34ba8a8, 0xe947aeae, 0xec41adad, 0xf451a5a5, 0xa2359797, 0xa7339494, 0xb52f9a9a, 0xb0299999, 0xa8399191, 0xba259f9f, 0xb62d9b9b, 0xab3b9090, 0xbf239c9c, 0xae3d9393, 0xad3f9292, 0xa1379696, 0xb32b9898, 0xb9279e9e, 0xbc219d9d, 0xa4319595, 0x392e1717, 0x3c281414, 0x2e341a1a, 0x2b321919, 0x33221111, 0x213e1f1f, 0x2d361b1b, 0x30201010, 0x24381c1c, 0x35261313, 0x36241212, 0x3a2c1616, 0x28301818, 0x223c1e1e, 0x273a1d1d, 0x3f2a1515, 0x02f5f7f7, 0x07f3f4f4, 0x15effafa, 0x10e9f9f9, 0x08f9f1f1, 0x1ae5ffff, 0x16edfbfb, 0x0bfbf0f0, 0x1fe3fcfc, 0x0efdf3f3, 0x0dfff2f2, 0x01f7f6f6, 0x13ebf8f8, 0x19e7fefe, 0x1ce1fdfd, 0x04f1f5f5, 0xc275b7b7, 0xc773b4b4, 0xd56fbaba, 0xd069b9b9, 0xc879b1b1, 0xda65bfbf, 0xd66dbbbb, 0xcb7bb0b0, 0xdf63bcbc, 0xce7db3b3, 0xcd7fb2b2, 0xc177b6b6, 0xd36bb8b8, 0xd967bebe, 0xdc61bdbd, 0xc471b5b5, 0x090e0707, 0x0c080404, 0x1e140a0a, 0x1b120909, 0x03020101, 0x111e0f0f, 0x1d160b0b, 0x00000000, 0x14180c0c, 0x05060303, 0x06040202, 0x0a0c0606, 0x18100808, 0x121c0e0e, 0x171a0d0d, 0x0f0a0505, 0x5295c7c7, 0x5793c4c4, 0x458fcaca, 0x4089c9c9, 0x5899c1c1, 0x4a85cfcf, 0x468dcbcb, 0x5b9bc0c0, 0x4f83cccc, 0x5e9dc3c3, 0x5d9fc2c2, 0x5197c6c6, 0x438bc8c8, 0x4987cece, 0x4c81cdcd, 0x5491c5c5, 0x596e3737, 0x5c683434, 0x4e743a3a, 0x4b723939, 0x53623131, 0x417e3f3f, 0x4d763b3b, 0x50603030, 0x44783c3c, 0x55663333, 0x56643232, 0x5a6c3636, 0x48703838, 0x427c3e3e, 0x477a3d3d, 0x5f6a3535, 0x694e2727, 0x6c482424, 0x7e542a2a, 0x7b522929, 0x63422121, 0x715e2f2f, 0x7d562b2b, 0x60402020, 0x74582c2c, 0x65462323, 0x66442222, 0x6a4c2626, 0x78502828, 0x725c2e2e, 0x775a2d2d, 0x6f4a2525, 0xa9ce6767, 0xacc86464, 0xbed46a6a, 0xbbd26969, 0xa3c26161, 0xb1de6f6f, 0xbdd66b6b, 0xa0c06060, 0xb4d86c6c, 0xa5c66363, 0xa6c46262, 0xaacc6666, 0xb8d06868, 0xb2dc6e6e, 0xb7da6d6d, 0xafca6565, 0x92158787, 0x97138484, 0x850f8a8a, 0x80098989, 0x98198181, 0x8a058f8f, 0x860d8b8b, 0x9b1b8080, 0x8f038c8c, 0x9e1d8383, 0x9d1f8282, 0x91178686, 0x830b8888, 0x89078e8e, 0x8c018d8d, 0x94118585, 0x32d5e7e7, 0x37d3e4e4, 0x25cfeaea, 0x20c9e9e9, 0x38d9e1e1, 0x2ac5efef, 0x26cdebeb, 0x3bdbe0e0, 0x2fc3ecec, 0x3edde3e3, 0x3ddfe2e2, 0x31d7e6e6, 0x23cbe8e8, 0x29c7eeee, 0x2cc1eded, 0x34d1e5e5, 0x62b5d7d7, 0x67b3d4d4, 0x75afdada, 0x70a9d9d9, 0x68b9d1d1, 0x7aa5dfdf, 0x76addbdb, 0x6bbbd0d0, 0x7fa3dcdc, 0x6ebdd3d3, 0x6dbfd2d2, 0x61b7d6d6, 0x73abd8d8, 0x79a7dede, 0x7ca1dddd, 0x64b1d5d5, 0xf9ae5757, 0xfca85454, 0xeeb45a5a, 0xebb25959, 0xf3a25151, 0xe1be5f5f, 0xedb65b5b, 0xf0a05050, 0xe4b85c5c, 0xf5a65353, 0xf6a45252, 0xfaac5656, 0xe8b05858, 0xe2bc5e5e, 0xe7ba5d5d, 0xffaa5555};
bit32 T2[256] = { 0x7799ee77, 0x749ce874, 0x7a8ef47a, 0x798bf279, 0x7193e271, 0x7f81fe7f, 0x7b8df67b, 0x7090e070, 0x7c84f87c, 0x7395e673, 0x7296e472, 0x769aec76, 0x7888f078, 0x7e82fc7e, 0x7d87fa7d, 0x759fea75, 0x47c98e47, 0x44cc8844, 0x4ade944a, 0x49db9249, 0x41c38241, 0x4fd19e4f, 0x4bdd964b, 0x40c08040, 0x4cd4984c, 0x43c58643, 0x42c68442, 0x46ca8c46, 0x48d89048, 0x4ed29c4e, 0x4dd79a4d, 0x45cf8a45, 0xa7f255a7, 0xa4f753a4, 0xaae54faa, 0xa9e049a9, 0xa1f859a1, 0xafea45af, 0xabe64dab, 0xa0fb5ba0, 0xacef43ac, 0xa3fe5da3, 0xa2fd5fa2, 0xa6f157a6, 0xa8e34ba8, 0xaee947ae, 0xadec41ad, 0xa5f451a5, 0x97a23597, 0x94a73394, 0x9ab52f9a, 0x99b02999, 0x91a83991, 0x9fba259f, 0x9bb62d9b, 0x90ab3b90, 0x9cbf239c, 0x93ae3d93, 0x92ad3f92, 0x96a13796, 0x98b32b98, 0x9eb9279e, 0x9dbc219d, 0x95a43195, 0x17392e17, 0x143c2814, 0x1a2e341a, 0x192b3219, 0x11332211, 0x1f213e1f, 0x1b2d361b, 0x10302010, 0x1c24381c, 0x13352613, 0x12362412, 0x163a2c16, 0x18283018, 0x1e223c1e, 0x1d273a1d, 0x153f2a15, 0xf702f5f7, 0xf407f3f4, 0xfa15effa, 0xf910e9f9, 0xf108f9f1, 0xff1ae5ff, 0xfb16edfb, 0xf00bfbf0, 0xfc1fe3fc, 0xf30efdf3, 0xf20dfff2, 0xf601f7f6, 0xf813ebf8, 0xfe19e7fe, 0xfd1ce1fd, 0xf504f1f5, 0xb7c275b7, 0xb4c773b4, 0xbad56fba, 0xb9d069b9, 0xb1c879b1, 0xbfda65bf, 0xbbd66dbb, 0xb0cb7bb0, 0xbcdf63bc, 0xb3ce7db3, 0xb2cd7fb2, 0xb6c177b6, 0xb8d36bb8, 0xbed967be, 0xbddc61bd, 0xb5c471b5, 0x07090e07, 0x040c0804, 0x0a1e140a, 0x091b1209, 0x01030201, 0x0f111e0f, 0x0b1d160b, 0x00000000, 0x0c14180c, 0x03050603, 0x02060402, 0x060a0c06, 0x08181008, 0x0e121c0e, 0x0d171a0d, 0x050f0a05, 0xc75295c7, 0xc45793c4, 0xca458fca, 0xc94089c9, 0xc15899c1, 0xcf4a85cf, 0xcb468dcb, 0xc05b9bc0, 0xcc4f83cc, 0xc35e9dc3, 0xc25d9fc2, 0xc65197c6, 0xc8438bc8, 0xce4987ce, 0xcd4c81cd, 0xc55491c5, 0x37596e37, 0x345c6834, 0x3a4e743a, 0x394b7239, 0x31536231, 0x3f417e3f, 0x3b4d763b, 0x30506030, 0x3c44783c, 0x33556633, 0x32566432, 0x365a6c36, 0x38487038, 0x3e427c3e, 0x3d477a3d, 0x355f6a35, 0x27694e27, 0x246c4824, 0x2a7e542a, 0x297b5229, 0x21634221, 0x2f715e2f, 0x2b7d562b, 0x20604020, 0x2c74582c, 0x23654623, 0x22664422, 0x266a4c26, 0x28785028, 0x2e725c2e, 0x2d775a2d, 0x256f4a25, 0x67a9ce67, 0x64acc864, 0x6abed46a, 0x69bbd269, 0x61a3c261, 0x6fb1de6f, 0x6bbdd66b, 0x60a0c060, 0x6cb4d86c, 0x63a5c663, 0x62a6c462, 0x66aacc66, 0x68b8d068, 0x6eb2dc6e, 0x6db7da6d, 0x65afca65, 0x87921587, 0x84971384, 0x8a850f8a, 0x89800989, 0x81981981, 0x8f8a058f, 0x8b860d8b, 0x809b1b80, 0x8c8f038c, 0x839e1d83, 0x829d1f82, 0x86911786, 0x88830b88, 0x8e89078e, 0x8d8c018d, 0x85941185, 0xe732d5e7, 0xe437d3e4, 0xea25cfea, 0xe920c9e9, 0xe138d9e1, 0xef2ac5ef, 0xeb26cdeb, 0xe03bdbe0, 0xec2fc3ec, 0xe33edde3, 0xe23ddfe2, 0xe631d7e6, 0xe823cbe8, 0xee29c7ee, 0xed2cc1ed, 0xe534d1e5, 0xd762b5d7, 0xd467b3d4, 0xda75afda, 0xd970a9d9, 0xd168b9d1, 0xdf7aa5df, 0xdb76addb, 0xd06bbbd0, 0xdc7fa3dc, 0xd36ebdd3, 0xd26dbfd2, 0xd661b7d6, 0xd873abd8, 0xde79a7de, 0xdd7ca1dd, 0xd564b1d5, 0x57f9ae57, 0x54fca854, 0x5aeeb45a, 0x59ebb259, 0x51f3a251, 0x5fe1be5f, 0x5bedb65b, 0x50f0a050, 0x5ce4b85c, 0x53f5a653, 0x52f6a452, 0x56faac56, 0x58e8b058, 0x5ee2bc5e, 0x5de7ba5d, 0x55ffaa55};
bit32 T3[256] = { 0x777799ee, 0x74749ce8, 0x7a7a8ef4, 0x79798bf2, 0x717193e2, 0x7f7f81fe, 0x7b7b8df6, 0x707090e0, 0x7c7c84f8, 0x737395e6, 0x727296e4, 0x76769aec, 0x787888f0, 0x7e7e82fc, 0x7d7d87fa, 0x75759fea, 0x4747c98e, 0x4444cc88, 0x4a4ade94, 0x4949db92, 0x4141c382, 0x4f4fd19e, 0x4b4bdd96, 0x4040c080, 0x4c4cd498, 0x4343c586, 0x4242c684, 0x4646ca8c, 0x4848d890, 0x4e4ed29c, 0x4d4dd79a, 0x4545cf8a, 0xa7a7f255, 0xa4a4f753, 0xaaaae54f, 0xa9a9e049, 0xa1a1f859, 0xafafea45, 0xababe64d, 0xa0a0fb5b, 0xacacef43, 0xa3a3fe5d, 0xa2a2fd5f, 0xa6a6f157, 0xa8a8e34b, 0xaeaee947, 0xadadec41, 0xa5a5f451, 0x9797a235, 0x9494a733, 0x9a9ab52f, 0x9999b029, 0x9191a839, 0x9f9fba25, 0x9b9bb62d, 0x9090ab3b, 0x9c9cbf23, 0x9393ae3d, 0x9292ad3f, 0x9696a137, 0x9898b32b, 0x9e9eb927, 0x9d9dbc21, 0x9595a431, 0x1717392e, 0x14143c28, 0x1a1a2e34, 0x19192b32, 0x11113322, 0x1f1f213e, 0x1b1b2d36, 0x10103020, 0x1c1c2438, 0x13133526, 0x12123624, 0x16163a2c, 0x18182830, 0x1e1e223c, 0x1d1d273a, 0x15153f2a, 0xf7f702f5, 0xf4f407f3, 0xfafa15ef, 0xf9f910e9, 0xf1f108f9, 0xffff1ae5, 0xfbfb16ed, 0xf0f00bfb, 0xfcfc1fe3, 0xf3f30efd, 0xf2f20dff, 0xf6f601f7, 0xf8f813eb, 0xfefe19e7, 0xfdfd1ce1, 0xf5f504f1, 0xb7b7c275, 0xb4b4c773, 0xbabad56f, 0xb9b9d069, 0xb1b1c879, 0xbfbfda65, 0xbbbbd66d, 0xb0b0cb7b, 0xbcbcdf63, 0xb3b3ce7d, 0xb2b2cd7f, 0xb6b6c177, 0xb8b8d36b, 0xbebed967, 0xbdbddc61, 0xb5b5c471, 0x0707090e, 0x04040c08, 0x0a0a1e14, 0x09091b12, 0x01010302, 0x0f0f111e, 0x0b0b1d16, 0x00000000, 0x0c0c1418, 0x03030506, 0x02020604, 0x06060a0c, 0x08081810, 0x0e0e121c, 0x0d0d171a, 0x05050f0a, 0xc7c75295, 0xc4c45793, 0xcaca458f, 0xc9c94089, 0xc1c15899, 0xcfcf4a85, 0xcbcb468d, 0xc0c05b9b, 0xcccc4f83, 0xc3c35e9d, 0xc2c25d9f, 0xc6c65197, 0xc8c8438b, 0xcece4987, 0xcdcd4c81, 0xc5c55491, 0x3737596e, 0x34345c68, 0x3a3a4e74, 0x39394b72, 0x31315362, 0x3f3f417e, 0x3b3b4d76, 0x30305060, 0x3c3c4478, 0x33335566, 0x32325664, 0x36365a6c, 0x38384870, 0x3e3e427c, 0x3d3d477a, 0x35355f6a, 0x2727694e, 0x24246c48, 0x2a2a7e54, 0x29297b52, 0x21216342, 0x2f2f715e, 0x2b2b7d56, 0x20206040, 0x2c2c7458, 0x23236546, 0x22226644, 0x26266a4c, 0x28287850, 0x2e2e725c, 0x2d2d775a, 0x25256f4a, 0x6767a9ce, 0x6464acc8, 0x6a6abed4, 0x6969bbd2, 0x6161a3c2, 0x6f6fb1de, 0x6b6bbdd6, 0x6060a0c0, 0x6c6cb4d8, 0x6363a5c6, 0x6262a6c4, 0x6666aacc, 0x6868b8d0, 0x6e6eb2dc, 0x6d6db7da, 0x6565afca, 0x87879215, 0x84849713, 0x8a8a850f, 0x89898009, 0x81819819, 0x8f8f8a05, 0x8b8b860d, 0x80809b1b, 0x8c8c8f03, 0x83839e1d, 0x82829d1f, 0x86869117, 0x8888830b, 0x8e8e8907, 0x8d8d8c01, 0x85859411, 0xe7e732d5, 0xe4e437d3, 0xeaea25cf, 0xe9e920c9, 0xe1e138d9, 0xefef2ac5, 0xebeb26cd, 0xe0e03bdb, 0xecec2fc3, 0xe3e33edd, 0xe2e23ddf, 0xe6e631d7, 0xe8e823cb, 0xeeee29c7, 0xeded2cc1, 0xe5e534d1, 0xd7d762b5, 0xd4d467b3, 0xdada75af, 0xd9d970a9, 0xd1d168b9, 0xdfdf7aa5, 0xdbdb76ad, 0xd0d06bbb, 0xdcdc7fa3, 0xd3d36ebd, 0xd2d26dbf, 0xd6d661b7, 0xd8d873ab, 0xdede79a7, 0xdddd7ca1, 0xd5d564b1, 0x5757f9ae, 0x5454fca8, 0x5a5aeeb4, 0x5959ebb2, 0x5151f3a2, 0x5f5fe1be, 0x5b5bedb6, 0x5050f0a0, 0x5c5ce4b8, 0x5353f5a6, 0x5252f6a4, 0x5656faac, 0x5858e8b0, 0x5e5ee2bc, 0x5d5de7ba, 0x5555ffaa};
bit8 S[16] = { 0x7, 0x4, 0xA, 0x9, 0x1, 0xF, 0xB, 0x0, 0xC, 0x3, 0x2, 0x6, 0x8, 0xE, 0xD, 0x5 };
bit8 S8[256] = { 0x77, 0x74, 0x7a, 0x79, 0x71, 0x7f, 0x7b, 0x70, 0x7c, 0x73, 0x72, 0x76, 0x78, 0x7e, 0x7d, 0x75, 0x47, 0x44, 0x4a, 0x49, 0x41, 0x4f, 0x4b, 0x40, 0x4c, 0x43, 0x42, 0x46, 0x48, 0x4e, 0x4d, 0x45, 0xa7, 0xa4, 0xaa, 0xa9, 0xa1, 0xaf, 0xab, 0xa0, 0xac, 0xa3, 0xa2, 0xa6, 0xa8, 0xae, 0xad, 0xa5, 0x97, 0x94, 0x9a, 0x99, 0x91, 0x9f, 0x9b, 0x90, 0x9c, 0x93, 0x92, 0x96, 0x98, 0x9e, 0x9d, 0x95, 0x17, 0x14, 0x1a, 0x19, 0x11, 0x1f, 0x1b, 0x10, 0x1c, 0x13, 0x12, 0x16, 0x18, 0x1e, 0x1d, 0x15, 0xf7, 0xf4, 0xfa, 0xf9, 0xf1, 0xff, 0xfb, 0xf0, 0xfc, 0xf3, 0xf2, 0xf6, 0xf8, 0xfe, 0xfd, 0xf5, 0xb7, 0xb4, 0xba, 0xb9, 0xb1, 0xbf, 0xbb, 0xb0, 0xbc, 0xb3, 0xb2, 0xb6, 0xb8, 0xbe, 0xbd, 0xb5, 0x7, 0x4, 0xa, 0x9, 0x1, 0xf, 0xb, 0x0, 0xc, 0x3, 0x2, 0x6, 0x8, 0xe, 0xd, 0x5, 0xc7, 0xc4, 0xca, 0xc9, 0xc1, 0xcf, 0xcb, 0xc0, 0xcc, 0xc3, 0xc2, 0xc6, 0xc8, 0xce, 0xcd, 0xc5, 0x37, 0x34, 0x3a, 0x39, 0x31, 0x3f, 0x3b, 0x30, 0x3c, 0x33, 0x32, 0x36, 0x38, 0x3e, 0x3d, 0x35, 0x27, 0x24, 0x2a, 0x29, 0x21, 0x2f, 0x2b, 0x20, 0x2c, 0x23, 0x22, 0x26, 0x28, 0x2e, 0x2d, 0x25, 0x67, 0x64, 0x6a, 0x69, 0x61, 0x6f, 0x6b, 0x60, 0x6c, 0x63, 0x62, 0x66, 0x68, 0x6e, 0x6d, 0x65, 0x87, 0x84, 0x8a, 0x89, 0x81, 0x8f, 0x8b, 0x80, 0x8c, 0x83, 0x82, 0x86, 0x88, 0x8e, 0x8d, 0x85, 0xe7, 0xe4, 0xea, 0xe9, 0xe1, 0xef, 0xeb, 0xe0, 0xec, 0xe3, 0xe2, 0xe6, 0xe8, 0xee, 0xed, 0xe5, 0xd7, 0xd4, 0xda, 0xd9, 0xd1, 0xdf, 0xdb, 0xd0, 0xdc, 0xd3, 0xd2, 0xd6, 0xd8, 0xde, 0xdd, 0xd5, 0x57, 0x54, 0x5a, 0x59, 0x51, 0x5f, 0x5b, 0x50, 0x5c, 0x53, 0x52, 0x56, 0x58, 0x5e, 0x5d, 0x55};
bit32 S8b[256] = { 0x7700, 0x7400, 0x7a00, 0x7900, 0x7100, 0x7f00, 0x7b00, 0x7000, 0x7c00, 0x7300, 0x7200, 0x7600, 0x7800, 0x7e00, 0x7d00, 0x7500, 0x4700, 0x4400, 0x4a00, 0x4900, 0x4100, 0x4f00, 0x4b00, 0x4000, 0x4c00, 0x4300, 0x4200, 0x4600, 0x4800, 0x4e00, 0x4d00, 0x4500, 0xa700, 0xa400, 0xaa00, 0xa900, 0xa100, 0xaf00, 0xab00, 0xa000, 0xac00, 0xa300, 0xa200, 0xa600, 0xa800, 0xae00, 0xad00, 0xa500, 0x9700, 0x9400, 0x9a00, 0x9900, 0x9100, 0x9f00, 0x9b00, 0x9000, 0x9c00, 0x9300, 0x9200, 0x9600, 0x9800, 0x9e00, 0x9d00, 0x9500, 0x1700, 0x1400, 0x1a00, 0x1900, 0x1100, 0x1f00, 0x1b00, 0x1000, 0x1c00, 0x1300, 0x1200, 0x1600, 0x1800, 0x1e00, 0x1d00, 0x1500, 0xf700, 0xf400, 0xfa00, 0xf900, 0xf100, 0xff00, 0xfb00, 0xf000, 0xfc00, 0xf300, 0xf200, 0xf600, 0xf800, 0xfe00, 0xfd00, 0xf500, 0xb700, 0xb400, 0xba00, 0xb900, 0xb100, 0xbf00, 0xbb00, 0xb000, 0xbc00, 0xb300, 0xb200, 0xb600, 0xb800, 0xbe00, 0xbd00, 0xb500, 0x700, 0x400, 0xa00, 0x900, 0x100, 0xf00, 0xb00, 0x0, 0xc00, 0x300, 0x200, 0x600, 0x800, 0xe00, 0xd00, 0x500, 0xc700, 0xc400, 0xca00, 0xc900, 0xc100, 0xcf00, 0xcb00, 0xc000, 0xcc00, 0xc300, 0xc200, 0xc600, 0xc800, 0xce00, 0xcd00, 0xc500, 0x3700, 0x3400, 0x3a00, 0x3900, 0x3100, 0x3f00, 0x3b00, 0x3000, 0x3c00, 0x3300, 0x3200, 0x3600, 0x3800, 0x3e00, 0x3d00, 0x3500, 0x2700, 0x2400, 0x2a00, 0x2900, 0x2100, 0x2f00, 0x2b00, 0x2000, 0x2c00, 0x2300, 0x2200, 0x2600, 0x2800, 0x2e00, 0x2d00, 0x2500, 0x6700, 0x6400, 0x6a00, 0x6900, 0x6100, 0x6f00, 0x6b00, 0x6000, 0x6c00, 0x6300, 0x6200, 0x6600, 0x6800, 0x6e00, 0x6d00, 0x6500, 0x8700, 0x8400, 0x8a00, 0x8900, 0x8100, 0x8f00, 0x8b00, 0x8000, 0x8c00, 0x8300, 0x8200, 0x8600, 0x8800, 0x8e00, 0x8d00, 0x8500, 0xe700, 0xe400, 0xea00, 0xe900, 0xe100, 0xef00, 0xeb00, 0xe000, 0xec00, 0xe300, 0xe200, 0xe600, 0xe800, 0xee00, 0xed00, 0xe500, 0xd700, 0xd400, 0xda00, 0xd900, 0xd100, 0xdf00, 0xdb00, 0xd000, 0xdc00, 0xd300, 0xd200, 0xd600, 0xd800, 0xde00, 0xdd00, 0xd500, 0x5700, 0x5400, 0x5a00, 0x5900, 0x5100, 0x5f00, 0x5b00, 0x5000, 0x5c00, 0x5300, 0x5200, 0x5600, 0x5800, 0x5e00, 0x5d00, 0x5500 };
bit32 S8c[256] = { 0x770000, 0x740000, 0x7a0000, 0x790000, 0x710000, 0x7f0000, 0x7b0000, 0x700000, 0x7c0000, 0x730000, 0x720000, 0x760000, 0x780000, 0x7e0000, 0x7d0000, 0x750000, 0x470000, 0x440000, 0x4a0000, 0x490000, 0x410000, 0x4f0000, 0x4b0000, 0x400000, 0x4c0000, 0x430000, 0x420000, 0x460000, 0x480000, 0x4e0000, 0x4d0000, 0x450000, 0xa70000, 0xa40000, 0xaa0000, 0xa90000, 0xa10000, 0xaf0000, 0xab0000, 0xa00000, 0xac0000, 0xa30000, 0xa20000, 0xa60000, 0xa80000, 0xae0000, 0xad0000, 0xa50000, 0x970000, 0x940000, 0x9a0000, 0x990000, 0x910000, 0x9f0000, 0x9b0000, 0x900000, 0x9c0000, 0x930000, 0x920000, 0x960000, 0x980000, 0x9e0000, 0x9d0000, 0x950000, 0x170000, 0x140000, 0x1a0000, 0x190000, 0x110000, 0x1f0000, 0x1b0000, 0x100000, 0x1c0000, 0x130000, 0x120000, 0x160000, 0x180000, 0x1e0000, 0x1d0000, 0x150000, 0xf70000, 0xf40000, 0xfa0000, 0xf90000, 0xf10000, 0xff0000, 0xfb0000, 0xf00000, 0xfc0000, 0xf30000, 0xf20000, 0xf60000, 0xf80000, 0xfe0000, 0xfd0000, 0xf50000, 0xb70000, 0xb40000, 0xba0000, 0xb90000, 0xb10000, 0xbf0000, 0xbb0000, 0xb00000, 0xbc0000, 0xb30000, 0xb20000, 0xb60000, 0xb80000, 0xbe0000, 0xbd0000, 0xb50000, 0x70000, 0x40000, 0xa0000, 0x90000, 0x10000, 0xf0000, 0xb0000, 0x0, 0xc0000, 0x30000, 0x20000, 0x60000, 0x80000, 0xe0000, 0xd0000, 0x50000, 0xc70000, 0xc40000, 0xca0000, 0xc90000, 0xc10000, 0xcf0000, 0xcb0000, 0xc00000, 0xcc0000, 0xc30000, 0xc20000, 0xc60000, 0xc80000, 0xce0000, 0xcd0000, 0xc50000, 0x370000, 0x340000, 0x3a0000, 0x390000, 0x310000, 0x3f0000, 0x3b0000, 0x300000, 0x3c0000, 0x330000, 0x320000, 0x360000, 0x380000, 0x3e0000, 0x3d0000, 0x350000, 0x270000, 0x240000, 0x2a0000, 0x290000, 0x210000, 0x2f0000, 0x2b0000, 0x200000, 0x2c0000, 0x230000, 0x220000, 0x260000, 0x280000, 0x2e0000, 0x2d0000, 0x250000, 0x670000, 0x640000, 0x6a0000, 0x690000, 0x610000, 0x6f0000, 0x6b0000, 0x600000, 0x6c0000, 0x630000, 0x620000, 0x660000, 0x680000, 0x6e0000, 0x6d0000, 0x650000, 0x870000, 0x840000, 0x8a0000, 0x890000, 0x810000, 0x8f0000, 0x8b0000, 0x800000, 0x8c0000, 0x830000, 0x820000, 0x860000, 0x880000, 0x8e0000, 0x8d0000, 0x850000, 0xe70000, 0xe40000, 0xea0000, 0xe90000, 0xe10000, 0xef0000, 0xeb0000, 0xe00000, 0xec0000, 0xe30000, 0xe20000, 0xe60000, 0xe80000, 0xee0000, 0xed0000, 0xe50000, 0xd70000, 0xd40000, 0xda0000, 0xd90000, 0xd10000, 0xdf0000, 0xdb0000, 0xd00000, 0xdc0000, 0xd30000, 0xd20000, 0xd60000, 0xd80000, 0xde0000, 0xdd0000, 0xd50000, 0x570000, 0x540000, 0x5a0000, 0x590000, 0x510000, 0x5f0000, 0x5b0000, 0x500000, 0x5c0000, 0x530000, 0x520000, 0x560000, 0x580000, 0x5e0000, 0x5d0000, 0x550000 };

__device__ bit32 arithmeticRightShift(bit32 x, bit32 n) { return (x >> n) | (x << (-n & 31)); }

__device__ bit32 arithmeticRightShiftBytePerm(bit32 x, bit32 n) { return __byte_perm(x, x, n); }
void gmix_column(unsigned char* r) {
    unsigned char a[4];
    unsigned char b[4];
    unsigned char c;
    unsigned char h;
    /* The array 'a' is simply a copy of the input array 'r'
     * The array 'b' is each element of the array 'a' multiplied by 2
     * in Rijndael's Galois field
     * a[n] ^ b[n] is element n multiplied by 3 in Rijndael's Galois field */
    for (c = 0; c < 4; c++) {
        a[c] = r[c];
        /* h is set to 0x01 if the high bit of r[c] is set, 0x00 otherwise */
        h = r[c] >> 7;    /* logical right shift, thus shifting in zeros */
        b[c] = r[c] << 1; /* implicitly removes high bit because b[c] is an 8-bit char, so we xor by 0x1b and not 0x11b in the next line */
        b[c] ^= h * 0x1B; /* Rijndael's Galois field */
    }
    r[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1]; /* 2 * a0 + a3 + a2 + 3 * a1 */
    r[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2]; /* 2 * a1 + a0 + a3 + 3 * a2 */
    r[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3]; /* 2 * a2 + a1 + a0 + 3 * a3 */
    r[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0]; /* 2 * a3 + a2 + a1 + 3 * a0 */
}
void calculate_tables() {
    bit8 r[4] = { 0 };
    bit32 result;
    for (bit32 i = 0; i < 256; i++) {
        r[0] = S[i & 0xf] ^ (S[i >> 4] << 4);
        r[1] = 0; r[2] = 0; r[3] = 0;
        gmix_column(r);
        result = (r[0]<<24) ^ (r[1] << 16) ^ (r[2] << 8) ^ (r[3] << 0);
        printf("0x%08x, ", result);
    }
    printf("\n");
    for (bit32 i = 0; i < 256; i++) {
        r[1] = S[i & 0xf] ^ (S[i >> 4] << 4);
        r[0] = 0; r[2] = 0; r[3] = 0;
        gmix_column(r);
        result = (r[0] << 24) ^ (r[1] << 16) ^ (r[2] << 8) ^ (r[3] << 0);
        printf("0x%08x, ", result);
    }
    printf("\n");
    for (bit32 i = 0; i < 256; i++) {
        r[2] = S[i & 0xf] ^ (S[i >> 4] << 4);
        r[0] = 0; r[1] = 0; r[3] = 0;
        gmix_column(r);
        result = (r[0] << 24) ^ (r[1] << 16) ^ (r[2] << 8) ^ (r[3] << 0);
        printf("0x%08x, ", result);
    }
    printf("\n");
    for (bit32 i = 0; i < 256; i++) {
        r[3] = S[i & 0xf] ^ (S[i >> 4] << 4);
        r[0] = 0; r[1] = 0; r[2] = 0;
        gmix_column(r);
        result = (r[0] << 24) ^ (r[1] << 16) ^ (r[2] << 8) ^ (r[3] << 0);
        printf("0x%08x, ", result);
    }
    printf("\n");
}
void generate_s8() {
    bit8 temp;
    for (int i = 0; i < 256; i++) {
        temp = S[i & 0xf] ^ (S[i >> 4] << 4);
        printf("0x%x, ",temp);
    }
}
void KLEIN64() {
	bit8 plaintext[8] = { 0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF }, input[8] = { 0x0 }, output[8] = { 0x0 };
    bit8 key[8] = { 0x00 }, temp[8] = { 0x0 };
	for (int i = 0; i < 8; i++) input [i] = plaintext[i];
     for (int j = 1; j <= 12; j++) {
        for (int i = 0; i < 8; i++) input[i] ^= key[i];
        for (int i = 0; i < 8; i++) { output[i] = S[input[i] & 0xf] ^ (S[input[i] >> 4] << 4); }
        for (int i = 0; i < 8; i++) input[(i+2)%8] = output[i];
        for (int i = 7; i >= 0; i--) printf("%02x", key[i]);    printf("\n");
        output[0] = input[3];
        output[1] = input[2];
        output[2] = input[1];
        output[3] = input[0];

        output[4] = input[7];
        output[5] = input[6];
        output[6] = input[5];
        output[7] = input[4];
        gmix_column(output);     gmix_column(output + 4);
        input[0] = output[3];
        input[1] = output[2];
        input[2] = output[1];
        input[3] = output[0];

        input[4] = output[7];
        input[5] = output[6];
        input[6] = output[5];
        input[7] = output[4];
 
        temp[0] = key[3];
        temp[1] = key[0];
        temp[2] = key[1];
        temp[3] = key[2];
        temp[4] = key[7];
        temp[5] = key[4];
        temp[6] = key[5];
        temp[7] = key[6];

        temp[4] ^= temp[0];
        temp[5] ^= temp[1];
        temp[6] ^= temp[2];
        temp[7] ^= temp[3];

        key[0] = temp[4];
        key[1] = temp[5];
        key[2] = temp[6];
        key[3] = temp[7];
        key[4] = temp[0];
        key[5] = temp[1];
        key[6] = temp[2];
        key[7] = temp[3];
        key[5] ^= j;
        key[1] = S[key[1] & 0xf] ^ (S[key[1] >> 4] << 4);
        key[2] = S[key[2] & 0xf] ^ (S[key[2] >> 4] << 4);
    }
    for (int i = 0; i < 8; i++) input[i] ^= key[i];
    for (int i = 7; i >=0; i--) printf("%02x",input[i]);    printf("\n");
}
void KLEIN64_table_based() {
    bit32 plaintext0 = 0xFFFFFFFF; // I image as the 64 bit is located as "plaintext1 plaintext0"
    bit32 plaintext1 = 0xFFFFFFFF;
    bit32 key0 = 0x0;
    bit32 key1 = 0x0;
    bit32 temp1 = 0x0, temp0 = 0x0;

    for (bit32 j = 1; j <= 12; j++) {
        temp1 = plaintext1 ^ key1;
        temp0 = plaintext0 ^ key0;

        plaintext0 = T3[(temp1 & 0x00FF0000) >> 16] ^ T2[(temp1 & 0xFF000000) >> 24] ^ T1[temp0 & 0x000000FF] ^ T0[(temp0 & 0x0000FF00) >> 8];
        plaintext1 = T3[(temp0 & 0x00FF0000) >> 16] ^ T2[(temp0 & 0xFF000000) >> 24] ^ T1[temp1 & 0x000000FF] ^ T0[(temp1 & 0x0000FF00) >> 8];
        printf("%08x%08x\n", key1, key0);

        key0 = (key0 << 8) ^ (key0 >> 24);
        key1 = (key1 << 8) ^ (key1 >> 24);
        key1 ^= key0;

        temp1 = key0;
        key0 = key1;
        key1 = temp1;

        key1 ^= (j << 8);
        key0 = (key0 & 0xFF0000FF) ^ (S[(key0 & 0x00000F00)>>8]<<8) ^ (S[(key0 & 0x0000F000) >> 12] << 12) ^ (S[(key0 & 0x000F0000) >> 16] << 16) ^ (S[(key0 & 0x00F00000) >> 20] << 20);
    }
    plaintext1 = plaintext1 ^ key1;
    plaintext0 = plaintext0 ^ key0;
    printf("%08x %08x\n",plaintext1, plaintext0);
}
__global__ void KLEIN64ExhaustiveSearch(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32 *T0G, bit32 *T1G, bit32 *T2G, bit32 *T3G, bit8* SG ) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ bit32 T0S[256];
    __shared__ bit32 T1S[256];
    __shared__ bit32 T2S[256];
    __shared__ bit32 T3S[256];
    __shared__ bit8 SS[16];
    if (threadIdx.x < 256) {
        if (threadIdx.x < 16) SS[threadIdx.x] = SG[threadIdx.x];
        T0S[threadIdx.x] = T0G[threadIdx.x];
        T1S[threadIdx.x] = T1G[threadIdx.x];
        T2S[threadIdx.x] = T2G[threadIdx.x];
        T3S[threadIdx.x] = T3G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1, plaintext1, plaintext0;
    bit32 key1 = threadIndex, key0 = 0;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    for (int i = 0; i < 1024*32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
        for (bit32 j = 1; j <= 12; j++) {
            temp1 = plaintext1 ^ key1;
            temp0 = plaintext0 ^ key0;

            plaintext0 = T3S[(temp1 & 0x00FF0000) >> 16] ^ T2S[(temp1 & 0xFF000000) >> 24] ^ T1S[temp0 & 0x000000FF] ^ T0S[(temp0 & 0x0000FF00) >> 8];
            plaintext1 = T3S[(temp0 & 0x00FF0000) >> 16] ^ T2S[(temp0 & 0xFF000000) >> 24] ^ T1S[temp1 & 0x000000FF] ^ T0S[(temp1 & 0x0000FF00) >> 8];

            key0 = (key0 << 8) ^ (key0 >> 24);
            key1 = (key1 << 8) ^ (key1 >> 24);
            key1 ^= key0;

            temp1 = key0;
            key0 = key1;
            key1 = temp1;

            key1 ^= (j << 8);
            key0 = (key0 & 0xFF0000FF) ^ (SS[(key0 & 0x00000F00) >> 8] << 8) ^ (SS[(key0 & 0x0000F000) >> 12] << 12) ^ (SS[(key0 & 0x000F0000) >> 16] << 16) ^ (SS[(key0 & 0x00F00000) >> 20] << 20);
        }
        plaintext1 = plaintext1 ^ key1;
        plaintext0 = plaintext0 ^ key0;
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08x%08x\n", threadIndex, i);
    }

}
__global__ void KLEIN64ExhaustiveSearch32Copies(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit32* T1G, bit32* T2G, bit32* T3G, bit8* SG) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit32 T1S[256];
    __shared__ bit32 T2S[256];
    __shared__ bit32 T3S[256];
    __shared__ bit8 SS[16];
    if (threadIdx.x < 256) {
        if (threadIdx.x < 16) SS[threadIdx.x] = SG[threadIdx.x];
        for (int i=0;i<32;i++) T0S[threadIdx.x][i] = T0G[threadIdx.x];
        T1S[threadIdx.x] = T1G[threadIdx.x];
        T2S[threadIdx.x] = T2G[threadIdx.x];
        T3S[threadIdx.x] = T3G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1,j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0, key1, key0;
    for (int i = 0; i < 1024*32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
        for (j = 1; j <= 12; j++) {
            temp1 = plaintext1 ^ key1;
            temp0 = plaintext0 ^ key0;

            plaintext0 = T3S[(temp1 & 0x00FF0000) >> 16] ^ T2S[(temp1 & 0xFF000000) >> 24] ^ T1S[temp0 & 0x000000FF] ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = T3S[(temp0 & 0x00FF0000) >> 16] ^ T2S[(temp0 & 0xFF000000) >> 24] ^ T1S[temp1 & 0x000000FF] ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];

            key0 = (key0 << 8) ^ (key0 >> 24);
            key1 = (key1 << 8) ^ (key1 >> 24);
            key1 ^= key0;

            temp1 = key0;
            key0 = key1;
            key1 = temp1;

            key1 ^= (j << 8);
            key0 = (key0 & 0xFF0000FF) ^ (SS[(key0 & 0x00000F00) >> 8] << 8) ^ (SS[(key0 & 0x0000F000) >> 12] << 12) ^ (SS[(key0 & 0x000F0000) >> 16] << 16) ^ (SS[(key0 & 0x00F00000) >> 20] << 20);
        }
        plaintext1 = plaintext1 ^ key1;
        plaintext0 = plaintext0 ^ key0;
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08x%08x\n", threadIndex, i);
    }
}
__global__ void KLEIN64ExhaustiveSearch32CopiesS8(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit32* T1G, bit32* T2G, bit32* T3G, bit8* SG) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit32 T1S[256];
    __shared__ bit32 T2S[256];
    __shared__ bit32 T3S[256];
    __shared__ bit8 SS[256];
    if (threadIdx.x < 256) {
        SS[threadIdx.x] = SG[threadIdx.x];
        for (int i = 0; i < 32; i++) T0S[threadIdx.x][i] = T0G[threadIdx.x];
        T1S[threadIdx.x] = T1G[threadIdx.x];
        T2S[threadIdx.x] = T2G[threadIdx.x];
        T3S[threadIdx.x] = T3G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1, j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0, key1, key0;
    for (int i = 0; i < 1024 * 32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
        for (j = 1; j <= 12; j++) {
            temp1 = plaintext1 ^ key1;
            temp0 = plaintext0 ^ key0;

            plaintext0 = T3S[(temp1 & 0x00FF0000) >> 16] ^ T2S[(temp1 & 0xFF000000) >> 24] ^ T1S[temp0 & 0x000000FF] ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = T3S[(temp0 & 0x00FF0000) >> 16] ^ T2S[(temp0 & 0xFF000000) >> 24] ^ T1S[temp1 & 0x000000FF] ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];

            key0 = (key0 << 8) ^ (key0 >> 24);
            key1 = (key1 << 8) ^ (key1 >> 24);
            key1 ^= key0;

            temp1 = key0;
            key0 = key1;
            key1 = temp1;

            key1 ^= (j << 8);
            key0 = (key0 & 0xFF0000FF) ^ (SS[(key0 & 0x0000FF00) >> 8] << 8) ^ (SS[(key0 & 0x00FF0000) >> 16] << 16) ;
        }
        plaintext1 = plaintext1 ^ key1;
        plaintext0 = plaintext0 ^ key0;
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08x%08x\n", threadIndex, i);
    }
}
__global__ void KLEIN64ExhaustiveSearch32CopiesS8SingleTable(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit8* SG) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit8 SS[256];
    if (threadIdx.x < 256) {
        SS[threadIdx.x] = SG[threadIdx.x];
        for (int i = 0; i < 32; i++) T0S[threadIdx.x][i] = T0G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1, j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0, key1, key0;

    for (int i = 0; i < 1024 * 32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
#pragma unroll
        for (j = 1; j <= 12; j++) {
            temp1 = plaintext1 ^ key1;
            temp0 = plaintext0 ^ key0;

            plaintext0 = arithmeticRightShift(T0S[(temp1 & 0x00FF0000) >> 16][warpThreadIndex],24) ^ arithmeticRightShift(T0S[(temp1 & 0xFF000000) >> 24][warpThreadIndex],16) ^ arithmeticRightShift(T0S[temp0 & 0x000000FF][warpThreadIndex],8) ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = arithmeticRightShift(T0S[(temp0 & 0x00FF0000) >> 16][warpThreadIndex],24) ^ arithmeticRightShift(T0S[(temp0 & 0xFF000000) >> 24][warpThreadIndex],16) ^ arithmeticRightShift(T0S[temp1 & 0x000000FF][warpThreadIndex],8) ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];

            key0 = arithmeticRightShift(key0, 24);
            key1 = arithmeticRightShift(key1, 24);
            key1 ^= key0;

            temp1 = key0;
            key0 = key1;
            key1 = temp1;

            key1 ^= (j << 8);
            key0 = (key0 & 0xFF0000FF) ^ (SS[(key0 & 0x0000FF00) >> 8] << 8) ^ (SS[(key0 & 0x00FF0000) >> 16] << 16);
        }
        plaintext1 = plaintext1 ^ key1;
        plaintext0 = plaintext0 ^ key0;
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08x%08x\n", threadIndex, i);
    }
}
__global__ void KLEIN64ExhaustiveSearch32CopiesS8SingleTableShift(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit32* SG1, bit32* SG2) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit32 S1[256];
    __shared__ bit32 S2[256];
    if (threadIdx.x < 256) {
        S1[threadIdx.x] = SG1[threadIdx.x];
        S2[threadIdx.x] = SG2[threadIdx.x];
        for (int i = 0; i < 32; i++) T0S[threadIdx.x][i] = T0G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1, j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0, key1, key0;

    for (int i = 0; i < 1024 * 32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
#pragma unroll
        for (j = 1; j <= 12; j++) {
            temp1 = plaintext1 ^ key1;
            temp0 = plaintext0 ^ key0;

            plaintext0 = arithmeticRightShift(T0S[(temp1 & 0x00FF0000) >> 16][warpThreadIndex], 24) ^ arithmeticRightShift(T0S[(temp1 & 0xFF000000) >> 24][warpThreadIndex], 16) ^ arithmeticRightShift(T0S[temp0 & 0x000000FF][warpThreadIndex], 8) ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = arithmeticRightShift(T0S[(temp0 & 0x00FF0000) >> 16][warpThreadIndex], 24) ^ arithmeticRightShift(T0S[(temp0 & 0xFF000000) >> 24][warpThreadIndex], 16) ^ arithmeticRightShift(T0S[temp1 & 0x000000FF][warpThreadIndex], 8) ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];

            key0 = arithmeticRightShift(key0, 24);
            key1 = arithmeticRightShift(key1, 24);
            key1 ^= key0;

            temp1 = key0;
            key0 = key1;
            key1 = temp1;

            key1 ^= (j << 8);
            key0 = (key0 & 0xFF0000FF) ^ (S1[(key0 & 0x0000FF00) >> 8]) ^ (S2[(key0 & 0x00FF0000) >> 16]);
        }
        plaintext1 = plaintext1 ^ key1;
        plaintext0 = plaintext0 ^ key0;
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08x%08x\n", threadIndex, i);
    }
}
__global__ void KLEIN64ExhaustiveSearch32CopiesS8SingleTableShiftBytePerm(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit32* SG1, bit32* SG2) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit32 S1[256];
    __shared__ bit32 S2[256];
    if (threadIdx.x < 256) {
        S1[threadIdx.x] = SG1[threadIdx.x];
        S2[threadIdx.x] = SG2[threadIdx.x];
        for (int i = 0; i < 32; i++) T0S[threadIdx.x][i] = T0G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1, j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0, key1, key0;

    for (int i = 0; i < 1024 * 32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
#pragma unroll
        for (j = 1; j <= 12; j++) {
            temp1 = plaintext1 ^ key1;
            temp0 = plaintext0 ^ key0;
            plaintext0 = arithmeticRightShiftBytePerm(T0S[(temp1 & 0x00FF0000) >> 16][warpThreadIndex], SHIFT_3_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[(temp1 & 0xFF000000) >> 24][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[temp0 & 0x000000FF][warpThreadIndex], SHIFT_1_RIGHT) ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = arithmeticRightShiftBytePerm(T0S[(temp0 & 0x00FF0000) >> 16][warpThreadIndex], SHIFT_3_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[(temp0 & 0xFF000000) >> 24][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[temp1 & 0x000000FF][warpThreadIndex], SHIFT_1_RIGHT) ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];
            key0 = arithmeticRightShiftBytePerm(key0, SHIFT_3_RIGHT);
            key1 = arithmeticRightShiftBytePerm(key1, SHIFT_3_RIGHT);
            key1 ^= key0;
            temp1 = key0;            key0 = key1;            key1 = temp1;
            key1 ^= (j << 8);
            key0 = (key0 & 0xFF0000FF) ^ (S1[arithmeticRightShiftBytePerm(key0 & 0x0000FF00, SHIFT_1_RIGHT)]) ^ (S2[arithmeticRightShiftBytePerm(key0 & 0x00FF0000, SHIFT_2_RIGHT)]);
        }
        plaintext1 = plaintext1 ^ key1;
        plaintext0 = plaintext0 ^ key0;
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08x%08x\n", threadIndex, i);
    }
}
__global__ void KLEIN96ExhaustiveSearch32CopiesS8SingleTable(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit8* SG) {
    bit64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit8 SS[256];
    if (threadIdx.x < 256) {
        SS[threadIdx.x] = SG[threadIdx.x];
        for (int i = 0; i < 32; i++) T0S[threadIdx.x][i] = T0G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1, j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0;
    bit64 key1, key0, temp2;

    for (bit64 i = 0; i < 1024 * 32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
#pragma unroll
        for (j = 1; j <= 20; j++) {
            temp1 = plaintext1 ^ (key1 >> 16);
            temp0 = plaintext0 ^ (key1 << 16) ^ (key0 >> 32);

            plaintext0 = arithmeticRightShiftBytePerm(T0S[(temp1 & 0x00FF0000) >> 16][warpThreadIndex], SHIFT_3_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[(temp1 & 0xFF000000) >> 24][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[temp0 & 0x000000FF][warpThreadIndex], SHIFT_1_RIGHT) ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = arithmeticRightShiftBytePerm(T0S[(temp0 & 0x00FF0000) >> 16][warpThreadIndex], SHIFT_3_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[(temp0 & 0xFF000000) >> 24][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[temp1 & 0x000000FF][warpThreadIndex], SHIFT_1_RIGHT) ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];


            key0 = ((key0 << 8) ^ (key0 >> 40)) & 0x0000FFFFFFFFFFFF;
            key1 = ((key1 << 8) ^ (key1 >> 40)) & 0x0000FFFFFFFFFFFF;
            key1 ^= key0;

            temp2 = key0;
            key0 = key1;
            key1 = temp2;

            key1 ^= (j << 24);
            key0 = (key0 & 0xFF0000FFFFFF) ^ (SS[(key0 & 0x0000FF000000) >> 24] << 24) ^ (bit64(SS[(key0 & 0x00FF00000000) >> 32]) << 32);
        }
        plaintext1 = plaintext1 ^ (key1>>16);
        plaintext0 = plaintext0 ^ (key1<<16) ^ (key0>>32);
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08llx%08llx\n", threadIndex, i);
    }
}
__global__ void KLEIN80ExhaustiveSearch32CopiesS8SingleTable(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit8* SG) {
    bit64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit8 SS[256];
    if (threadIdx.x < 256) {
        SS[threadIdx.x] = SG[threadIdx.x];
        for (int i = 0; i < 32; i++) T0S[threadIdx.x][i] = T0G[threadIdx.x];
    }
    __syncthreads();
    bit32 temp0, temp1, j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0;
    bit64 key1, key0, temp2;

    for (bit64 i = 0; i < 1024 * 32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
#pragma unroll
        for (j = 1; j <= 16; j++) {
            temp1 = plaintext1 ^ (key1 >> 8);
            temp0 = plaintext0 ^ (key1 << 24) ^ (key0 >> 16);

 //           plaintext0 = arithmeticRightShift(T0S[(temp1 & 0x00FF0000) >> 16][warpThreadIndex], 24) ^ arithmeticRightShift(T0S[(temp1 & 0xFF000000) >> 24][warpThreadIndex], 16) ^ arithmeticRightShift(T0S[temp0 & 0x000000FF][warpThreadIndex], 8) ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
 //           plaintext1 = arithmeticRightShift(T0S[(temp0 & 0x00FF0000) >> 16][warpThreadIndex], 24) ^ arithmeticRightShift(T0S[(temp0 & 0xFF000000) >> 24][warpThreadIndex], 16) ^ arithmeticRightShift(T0S[temp1 & 0x000000FF][warpThreadIndex], 8) ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext0 = arithmeticRightShiftBytePerm(T0S[(temp1 & 0x00FF0000) >> 16][warpThreadIndex], SHIFT_3_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[(temp1 & 0xFF000000) >> 24][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[temp0 & 0x000000FF][warpThreadIndex], SHIFT_1_RIGHT) ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = arithmeticRightShiftBytePerm(T0S[(temp0 & 0x00FF0000) >> 16][warpThreadIndex], SHIFT_3_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[(temp0 & 0xFF000000) >> 24][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(T0S[temp1 & 0x000000FF][warpThreadIndex], SHIFT_1_RIGHT) ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];

            key0 = ((key0 << 8) ^ (key0 >> 32)) & 0x000000FFFFFFFFFF;
            key1 = ((key1 << 8) ^ (key1 >> 32)) & 0x000000FFFFFFFFFF;
            key1 ^= key0;

            temp2 = key0;
            key0 = key1;
            key1 = temp2;

            key1 ^= (j << 16);
            key0 = (key0 & 0xFF0000FFFF) ^ (SS[(key0 & 0x0000FF0000) >> 16] << 16) ^ (SS[(key0 & 0x00FF000000) >> 24] << 24);
        }
        plaintext1 = plaintext1 ^ (key1 >> 8);
        plaintext0 = plaintext0 ^ (key1 << 24) ^ (key0 >> 16);
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08llx%08llx\n", threadIndex, i);
    }
}
__global__ void KLEIN64ExhaustiveSearch32CopiesS8SingleTableS(bit32 pt1, bit32 pt0, bit32 ct1, bit32 ct0, bit32* T0G, bit8* SG) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int warpThreadIndex = threadIdx.x & 31;
    __shared__ bit32 T0S[256][32];
    __shared__ bit8 SS[64][32][4];
    if (threadIdx.x < 256) {
        for (int i = 0; i < 32; i++) {
            T0S[threadIdx.x][i] = T0G[threadIdx.x];
            SS[threadIdx.x / 4][i][threadIdx.x % 4] = SG[threadIdx.x];
        }
    }
    __syncthreads();
    bit32 temp0, temp1, j;
    bit32 ciphertext1 = ct1;
    bit32 ciphertext0 = ct0;
    bit32 plaintext1, plaintext0, key1, key0;
    for (int i = 0; i < 1024 * 32; i++) {
        plaintext1 = pt1;
        plaintext0 = pt0;
        key1 = threadIndex;
        key0 = i;
        for (j = 1; j <= 12; j++) {
            temp1 = plaintext1 ^ key1;
            temp0 = plaintext0 ^ key0;

            plaintext0 = arithmeticRightShift(T0S[(temp1 & 0x00FF0000) >> 16][warpThreadIndex], 24) ^ arithmeticRightShift(T0S[(temp1 & 0xFF000000) >> 24][warpThreadIndex], 16) ^ arithmeticRightShift(T0S[temp0 & 0x000000FF][warpThreadIndex], 8) ^ T0S[(temp0 & 0x0000FF00) >> 8][warpThreadIndex];
            plaintext1 = arithmeticRightShift(T0S[(temp0 & 0x00FF0000) >> 16][warpThreadIndex], 24) ^ arithmeticRightShift(T0S[(temp0 & 0xFF000000) >> 24][warpThreadIndex], 16) ^ arithmeticRightShift(T0S[temp1 & 0x000000FF][warpThreadIndex], 8) ^ T0S[(temp1 & 0x0000FF00) >> 8][warpThreadIndex];

            key0 = (key0 << 8) ^ (key0 >> 24);
            key1 = (key1 << 8) ^ (key1 >> 24);
            key1 ^= key0;

            temp1 = key0;
            key0 = key1;
            key1 = temp1;

            key1 ^= (j << 8);
            key0 = (key0 & 0xFF0000FF) ^ (SS[((key0 & 0x0000FF00) >> 8)/4][warpThreadIndex][((key0 & 0x0000FF00) >> 8) % 4] << 8) ^ (SS[((key0 & 0x00FF0000) >> 16)/4][warpThreadIndex][((key0 & 0x00FF0000) >> 16) % 4] << 16);
        }
        plaintext1 = plaintext1 ^ key1;
        plaintext0 = plaintext0 ^ key0;
        if (plaintext1 == ciphertext1)
            if (plaintext0 == ciphertext0)
                printf("The secret key is %08x%08x\n", threadIndex, i);
    }
}
void ExhaustiveSearch() {
    bit32 plaintext0 = 0xFFFFFFFF; // I imagine as the 64 bit is located as "plaintext1 plaintext0"
    bit32 plaintext1 = 0xFFFFFFFF;
    bit32 ciphertext0 = 0x14722bbe; 
    bit32 ciphertext1 = 0xcdc0b51f;
    bit32 ciphertext2 = 0x3D8E8E36;
    bit32 ciphertext3 = 0xDB9FA7D3;
    bit32 ciphertext4 = 0x1A53A431;
    bit32 ciphertext5 = 0x6677E20D;

    // Allocate Tables
    bit32* t0, * t1, * t2, * t3, *s8b, *s8c;    bit8* s4; bit8* s8;
    cudaMallocManaged(&t0, 256 * sizeof(bit32));
    cudaMallocManaged(&t1, 256 * sizeof(bit32));
    cudaMallocManaged(&t2, 256 * sizeof(bit32));
    cudaMallocManaged(&t3, 256 * sizeof(bit32));
    cudaMallocManaged(&s8b, 256 * sizeof(bit32));
    cudaMallocManaged(&s8c, 256 * sizeof(bit32));
    cudaMallocManaged(&s4, 16 * sizeof(bit8));
    cudaMallocManaged(&s8, 256 * sizeof(bit8));
    for (int i = 0; i < 256; i++) {
        t0[i] = T0[i];
        t1[i] = T1[i];
        t2[i] = T2[i];
        t3[i] = T3[i];
        s8[i] = S8[i];
        s8b[i] = S8b[i];
        s8c[i] = S8c[i];
    }
    for (int i = 0; i < 16; i++) s4[i] = S[i];

    clock_t beginTime = clock();
//    KLEIN64ExhaustiveSearch << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext1, ciphertext0,t0,t1,t2,t3,s4);
//    KLEIN64ExhaustiveSearch32Copies << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext1, ciphertext0, t0, t1, t2, t3, s4);
//    KLEIN64ExhaustiveSearch32CopiesS8 << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext1, ciphertext0, t0, t1, t2, t3, s8);
//    KLEIN64ExhaustiveSearch32CopiesS8SingleTable << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext1, ciphertext0, t0, s8);
//    KLEIN64ExhaustiveSearch32CopiesS8SingleTableShift << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext1, ciphertext0, t0, s8b, s8c);
//    KLEIN64ExhaustiveSearch32CopiesS8SingleTableShiftBytePerm << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext1, ciphertext0, t0, s8b, s8c);
      KLEIN80ExhaustiveSearch32CopiesS8SingleTable << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext5, ciphertext4, t0, s8);
//    KLEIN96ExhaustiveSearch32CopiesS8SingleTable << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext3, ciphertext2, t0, s8);
//    KLEIN64ExhaustiveSearch32CopiesS8SingleTableS << <BLOCKS, THREADS >> > (plaintext1, plaintext0, ciphertext1, ciphertext0, t0, s8);
    cudaDeviceSynchronize();
    printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
    printf("-------------------------------\n");

    cudaFree(t0);    cudaFree(t1);    cudaFree(t2);    cudaFree(t3);    cudaFree(s4); cudaFree(s8); cudaFree(s8b); cudaFree(s8c);
}
void sboxshift() {
    for (int i = 0; i < 256; i++) printf("0x%x, ", S8[i] << 16); printf("\n");
}
int main(void) {
	cudaSetDevice(0);
//	KLEIN64();   printf("\n\n");
//  KLEIN64_table_based();
//  calculate_tables();
    ExhaustiveSearch();
//     sboxshift();
//    generate_s8();
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

