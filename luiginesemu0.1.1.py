#!python3.15
"""luigines emu 1.x — single-file NES emulator (NTSC 2A03 / 2C02).

1.x changes vs 0.1.x
  CPU   all 256 opcodes (official + unofficial + JAM), generated straight-line
        handlers, zero-page/stack direct RAM access, RMW dummy writes,
        level-triggered IRQ line, exact cycle counts incl. page-cross/branch.
  PPU   loopy v/t/x scrolling with mid-scanline split (dot 256/257), correct
        sprite Y (+1), OAM priority, behind-BG quirk, real sprite-0 hit,
        left-8 clipping, greyscale/emphasis, odd-frame dot skip, per-line
        background cache.
  APU   frame sequencer in CPU cycles (0.1.x ran it at 2x speed), lazy
        catch-up sync, int16 PCM blocks instead of per-sample queue pushes.
  GUI   1 ms Windows timer resolution, hybrid sleep/spin pacing, audio-buffer
        rate control (no long-term drift), C-speed framebuffer blit.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import sys
import threading
import time
import tkinter as tk
from array import array
from tkinter import filedialog, messagebox

# =============================================================================
#  PALETTE - 64 NES colors (RGB) blue-tinted per request
# =============================================================================

_NES_PAL = (
    (0x66,0x66,0x66),(0x00,0x2A,0x88),(0x14,0x12,0xA7),(0x3B,0x00,0xA4),
    (0x5C,0x00,0x7E),(0x6E,0x00,0x40),(0x6C,0x06,0x00),(0x56,0x1D,0x00),
    (0x33,0x35,0x00),(0x0B,0x48,0x00),(0x00,0x52,0x00),(0x00,0x4F,0x08),
    (0x00,0x40,0x4D),(0x00,0x00,0x00),(0x00,0x00,0x00),(0x00,0x00,0x00),
    (0xAD,0xAD,0xAD),(0x15,0x5F,0xD9),(0x42,0x40,0xFF),(0x75,0x27,0xFE),
    (0xA0,0x1A,0xCC),(0xB7,0x1E,0x7B),(0xB5,0x31,0x20),(0x99,0x4E,0x00),
    (0x6B,0x6D,0x00),(0x38,0x87,0x00),(0x0C,0x93,0x00),(0x00,0x8F,0x32),
    (0x00,0x7C,0x8D),(0x00,0x00,0x00),(0x00,0x00,0x00),(0x00,0x00,0x00),
    (0xFF,0xFE,0xFF),(0x64,0xB0,0xFF),(0x92,0x90,0xFF),(0xC6,0x76,0xFF),
    (0xF3,0x6A,0xFF),(0xFE,0x6E,0xCC),(0xFE,0x81,0x70),(0xEA,0x9E,0x22),
    (0xBC,0xBE,0x00),(0x88,0xD8,0x00),(0x5C,0xE4,0x30),(0x45,0xE0,0x82),
    (0x48,0xCD,0xDE),(0x4F,0x4F,0x4F),(0x00,0x00,0x00),(0x00,0x00,0x00),
    (0xFF,0xFE,0xFF),(0xC0,0xDF,0xFF),(0xD3,0xD2,0xFF),(0xE8,0xC8,0xFF),
    (0xFB,0xC2,0xFF),(0xFE,0xC4,0xEA),(0xFE,0xCC,0xC5),(0xF7,0xD8,0xA5),
    (0xE4,0xE5,0x94),(0xCF,0xEF,0x96),(0xBD,0xF4,0xAB),(0xB3,0xF3,0xCC),
    (0xB5,0xEB,0xF2),(0xB8,0xB8,0xB8),(0x00,0x00,0x00),(0x00,0x00,0x00),
)


def _blue_tint(rgb):
    r, g, b = rgb
    r = int(r * 0.55)
    g = int(g * 0.75)
    b = min(255, int(b * 1.10) + 35)
    return (r, g, b)


NES_PALETTE_RGB = tuple(_NES_PAL)  # authentic NES colors, classic MarioNES-style
NES_PALETTE_HEX = tuple("#%02x%02x%02x" % c for c in NES_PALETTE_RGB)


def _build_rgb_planes():
    """8 emphasis variants x (R, G, B) 256-byte translate tables."""
    planes = []
    for emph in range(8):
        rs, gs, bs = bytearray(256), bytearray(256), bytearray(256)
        for i in range(256):
            r, g, b = NES_PALETTE_RGB[i & 0x3F]
            if emph & 1:          # $2001 bit 5: emphasise red
                g *= 0.816; b *= 0.816
            if emph & 2:          # bit 6: green
                r *= 0.816; b *= 0.816
            if emph & 4:          # bit 7: blue
                r *= 0.816; g *= 0.816
            rs[i], gs[i], bs[i] = int(r), int(g), int(b)
        planes.append((bytes(rs), bytes(gs), bytes(bs)))
    return tuple(planes)


RGB_PLANES = _build_rgb_planes()


# =============================================================================
#  iNES / NES 2.0 loader
# =============================================================================

class INESHeader:
    __slots__ = ("prg_banks", "chr_banks", "mapper", "mirroring", "battery",
                 "trainer", "four_screen", "is_nes2", "submapper", "prg_ram",
                 "tv_system", "filename", "title")

    def __init__(self, data, filename=""):
        if len(data) < 16 or data[0:4] != b"NES\x1a":
            raise ValueError("Not a valid iNES file (missing NES<EOF> magic)")
        self.filename = filename
        self.title = os.path.basename(filename) if filename else "untitled"
        self.prg_banks = data[4]
        self.chr_banks = data[5]
        f6, f7 = data[6], data[7]
        self.mirroring = "vertical" if (f6 & 1) else "horizontal"
        self.battery = bool(f6 & 2)
        self.trainer = bool(f6 & 4)
        self.four_screen = bool(f6 & 8)
        self.is_nes2 = ((f7 >> 2) & 3) == 2
        if self.is_nes2:
            self.mapper = (f6 >> 4) | (f7 & 0xF0) | ((data[8] & 0x0F) << 8)
            self.submapper = (data[8] >> 4) & 0x0F
            prg_hi = data[9] & 0x0F
            chr_hi = (data[9] >> 4) & 0x0F
            if prg_hi != 0xF:
                self.prg_banks |= (prg_hi << 8)
            if chr_hi != 0xF:
                self.chr_banks |= (chr_hi << 8)
            self.prg_ram = (64 << (data[10] & 0x0F)) if (data[10] & 0x0F) else 0
            self.tv_system = "PAL" if (data[12] & 1) else "NTSC"
        else:
            # "DiskDude!"-style junk in bytes 7..15 corrupts the mapper high
            # nibble; real iNES 1.0 dumps keep bytes 12..15 zero.
            hi = f7 & 0xF0 if not any(data[12:16]) else 0
            self.mapper = (f6 >> 4) | hi
            self.submapper = 0
            self.prg_ram = data[8] * 8192 if data[8] else 8192
            self.tv_system = "PAL" if (data[9] & 1) else "NTSC"
        if self.prg_banks == 0:
            raise ValueError("ROM header declares 0 PRG banks")


class ROM:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        self.header = INESHeader(data, path)
        off = 16
        if self.header.trainer:
            self.trainer = data[off:off + 512]
            off += 512
        else:
            self.trainer = b""
        prg_size = self.header.prg_banks * 0x4000
        chr_size = self.header.chr_banks * 0x2000
        self.prg = bytearray(data[off:off + prg_size])
        if len(self.prg) < prg_size:
            raise ValueError("Truncated ROM: PRG data is incomplete")
        off += prg_size
        if chr_size > 0:
            self.chr = bytearray(data[off:off + chr_size])
            if len(self.chr) < chr_size:
                raise ValueError("Truncated ROM: CHR data is incomplete")
        else:
            self.chr = bytearray(0x2000)
        self.has_chr_ram = chr_size == 0


# =============================================================================
#  Mappers - 0, 1, 2, 3, 4, 7, 11, 66, 71
#  Every mapper exposes prg_map (4 x 8 KB slots) and chr_map (8 x 1 KB slots)
#  as byte offsets.  The bus and PPU index those lists directly, so the lists
#  are always mutated in place, never rebound.
# =============================================================================

class Mapper:
    def __init__(self, rom):
        self.rom = rom
        self.prg = rom.prg
        self.chr = rom.chr
        self.chr_is_ram = rom.has_chr_ram
        self.prg_len = len(rom.prg)
        self.chr_len = len(rom.chr)
        self.prg_map = [0, 0, 0, 0]
        self.chr_map = [0] * 8
        h = rom.header
        self.four_screen = h.four_screen
        self.mirroring = "four-screen" if h.four_screen else h.mirroring
        self.prg_ram = bytearray(0x2000)
        if rom.trainer:
            self.prg_ram[0x1000:0x1200] = rom.trainer
        self.cpu = None
        self.set_prg_16k(0, 0)
        self.set_prg_16k(1, -1)
        self.set_chr_8k(0)

    # -- banking helpers ---------------------------------------------------
    def set_prg_8k(self, slot, bank):
        self.prg_map[slot] = (bank % (self.prg_len >> 13)) << 13

    def set_prg_16k(self, slot, bank):
        b = (bank % (self.prg_len >> 14)) << 14
        self.prg_map[slot * 2] = b
        self.prg_map[slot * 2 + 1] = (b + 0x2000) % self.prg_len

    def set_prg_32k(self, bank):
        self.set_prg_16k(0, bank * 2)
        self.set_prg_16k(1, bank * 2 + 1)

    def set_chr_1k(self, slot, bank):
        self.chr_map[slot] = (bank % (self.chr_len >> 10)) << 10

    def set_chr_4k(self, slot, bank):
        b = (bank % max(1, self.chr_len >> 12)) << 12
        for i in range(4):
            self.chr_map[slot * 4 + i] = (b + (i << 10)) % self.chr_len

    def set_chr_8k(self, bank):
        b = (bank % max(1, self.chr_len >> 13)) << 13
        for i in range(8):
            self.chr_map[i] = (b + (i << 10)) % self.chr_len

    def set_mirroring(self, m):
        if not self.four_screen:
            self.mirroring = m

    # -- CPU side ($4020-$7FFF reads; $4020-$FFFF writes) ------------------
    def cpu_read(self, addr):
        if addr >= 0x6000:
            return self.prg_ram[addr & 0x1FFF]
        return addr >> 8                      # open bus approximation

    def cpu_write(self, addr, val):
        if addr >= 0x8000:
            self.write_reg(addr, val)
        elif addr >= 0x6000:
            self.prg_ram[addr & 0x1FFF] = val

    def write_reg(self, addr, val):
        pass

    def scanline_tick(self):
        pass

    def a12_rise(self):
        pass


class NROM(Mapper):
    pass


class MMC1(Mapper):
    def __init__(self, rom):
        super().__init__(rom)
        self.shift = 0x10
        self.control = 0x0C
        self.chr0 = 0
        self.chr1 = 0
        self.prg_bank = 0
        self._last_write = -1
        self._surom = self.prg_len >= 0x80000
        self._update()

    def write_reg(self, addr, val):
        # The MMC1 ignores a write on the cycle right after another write.
        # RMW instructions (INC $8000 etc.) write twice inside one
        # instruction; only the first (original-value) write counts.
        cyc = self.cpu.cycles if self.cpu is not None else -2
        if cyc == self._last_write or cyc == self._last_write + 1:
            return
        self._last_write = cyc
        if val & 0x80:
            self.shift = 0x10
            self.control |= 0x0C
            self._update()
            return
        complete = self.shift & 1
        self.shift = (self.shift >> 1) | ((val & 1) << 4)
        if complete:
            reg = (addr >> 13) & 3
            v = self.shift
            self.shift = 0x10
            if reg == 0:
                self.control = v
            elif reg == 1:
                self.chr0 = v
            elif reg == 2:
                self.chr1 = v
            else:
                self.prg_bank = v
            self._update()

    def _update(self):
        self.set_mirroring(("one-screen-lo", "one-screen-hi",
                            "vertical", "horizontal")[self.control & 3])
        mode = (self.control >> 2) & 3
        outer = (self.chr0 & 0x10) if self._surom else 0
        bank = self.prg_bank & 0x0F
        if mode < 2:
            self.set_prg_32k((outer | (bank & 0x0E)) >> 1)
        elif mode == 2:
            self.set_prg_16k(0, outer)
            self.set_prg_16k(1, outer | bank)
        else:
            self.set_prg_16k(0, outer | bank)
            self.set_prg_16k(1, (outer | 0x0F) if self._surom else -1)
        if self.control & 0x10:
            self.set_chr_4k(0, self.chr0)
            self.set_chr_4k(1, self.chr1)
        else:
            self.set_chr_8k(self.chr0 >> 1)


class UNROM(Mapper):
    def write_reg(self, addr, val):
        self.set_prg_16k(0, val & 0x0F)


class CNROM(Mapper):
    def write_reg(self, addr, val):
        self.set_chr_8k(val)


class AOROM(Mapper):
    def __init__(self, rom):
        super().__init__(rom)
        self.set_prg_32k(0)
        self.set_mirroring("one-screen-lo")

    def write_reg(self, addr, val):
        self.set_prg_32k(val & 0x07)
        self.set_mirroring("one-screen-hi" if (val & 0x10) else "one-screen-lo")


class GxROM(Mapper):
    def __init__(self, rom):
        super().__init__(rom)
        self.set_prg_32k(0)

    def write_reg(self, addr, val):
        self.set_prg_32k((val >> 4) & 3)
        self.set_chr_8k(val & 3)


class ColorDreams(Mapper):
    def __init__(self, rom):
        super().__init__(rom)
        self.set_prg_32k(0)

    def write_reg(self, addr, val):
        self.set_prg_32k(val & 3)
        self.set_chr_8k(val >> 4)


class Camerica(Mapper):
    def write_reg(self, addr, val):
        if addr >= 0xC000:
            self.set_prg_16k(0, val & 0x0F)
        elif 0x9000 <= addr < 0xA000 and self.rom.header.submapper == 1:
            # Fire Hawk single-screen select
            self.set_mirroring("one-screen-hi" if (val & 0x10) else "one-screen-lo")


class MMC3(Mapper):
    def __init__(self, rom):
        super().__init__(rom)
        self.bank_select = 0
        self.regs = [0, 2, 4, 5, 6, 7, 0, 1]
        self.irq_latch = 0
        self.irq_counter = 0
        self.irq_reload = False
        self.irq_enable = False
        self._update()

    def _update(self):
        r = self.regs
        if self.bank_select & 0x40:
            self.set_prg_8k(0, -2)
            self.set_prg_8k(2, r[6])
        else:
            self.set_prg_8k(0, r[6])
            self.set_prg_8k(2, -2)
        self.set_prg_8k(1, r[7])
        self.set_prg_8k(3, -1)
        c = self.set_chr_1k
        lo = 4 if (self.bank_select & 0x80) else 0
        hi = 0 if lo else 4
        c(lo + 0, r[0] & 0xFE); c(lo + 1, r[0] | 1)
        c(lo + 2, r[1] & 0xFE); c(lo + 3, r[1] | 1)
        c(hi + 0, r[2]); c(hi + 1, r[3]); c(hi + 2, r[4]); c(hi + 3, r[5])

    def write_reg(self, addr, val):
        even = not (addr & 1)
        if addr < 0xA000:
            if even:
                self.bank_select = val
            else:
                self.regs[self.bank_select & 7] = val
            self._update()
        elif addr < 0xC000:
            if even:
                self.set_mirroring("horizontal" if (val & 1) else "vertical")
        elif addr < 0xE000:
            if even:
                self.irq_latch = val
            else:
                self.irq_counter = 0
                self.irq_reload = True
        else:
            self.irq_enable = not even
            if even and self.cpu is not None:
                self.cpu.irq &= ~1            # acknowledge

    def scanline_tick(self):
        if self.irq_counter == 0 or self.irq_reload:
            self.irq_counter = self.irq_latch
            self.irq_reload = False
        else:
            self.irq_counter -= 1
        if self.irq_counter == 0 and self.irq_enable and self.cpu is not None:
            self.cpu.irq |= 1

    a12_rise = scanline_tick


_MAPPER_TABLE = {
    0: NROM, 1: MMC1, 2: UNROM, 3: CNROM, 4: MMC3,
    7: AOROM, 11: ColorDreams, 66: GxROM, 71: Camerica,
}


def make_mapper(rom):
    cls = _MAPPER_TABLE.get(rom.header.mapper)
    if cls is None:
        raise ValueError(
            f"Unsupported mapper {rom.header.mapper}. Supported: "
            + ", ".join(str(m) for m in sorted(_MAPPER_TABLE)))
    return cls(rom)


# =============================================================================
#  Timing constants
# =============================================================================

AUDIO_FILES = False
AUDIO_ENABLED = True
SAMPLE_RATE = 48000
PPU_SCANLINES = 262
PPU_DOTS_PER_LINE = 341

TARGET_FPS = 60.0988
FRAME_SEC = 1.0 / TARGET_FPS
CPU_CYCLES_PER_LINE = PPU_DOTS_PER_LINE / 3.0          # 113.667
CPU_CYCLES_AT_DOT256 = 256 / 3.0                       # 85.333
CPU_CYCLES_AT_DOT260 = 260 / 3.0
CPU_CYCLES_AT_DOT324 = 324 / 3.0
# 89342 dots even frames, 89341 odd frames (rendering on) -> 29780.5 avg.
CPU_CYCLES_PER_FRAME = (PPU_SCANLINES * PPU_DOTS_PER_LINE - 0.5) / 3.0
# Sample clock locked to the emulated CPU so 60.00 FPS video always yields
# exactly 800 samples/frame; the GUI's rate control absorbs device drift.
CYCLES_PER_SAMPLE = CPU_CYCLES_PER_FRAME * TARGET_FPS / SAMPLE_RATE


# =============================================================================
#  PPU (2C02)
# =============================================================================

_MIRROR_PAGES = {
    "horizontal": (0, 0, 1, 1),
    "vertical": (0, 1, 0, 1),
    "one-screen-lo": (0, 0, 0, 0),
    "one-screen-hi": (1, 1, 1, 1),
    "four-screen": (0, 1, 2, 3),
}

_PATTERN_ROW_CACHE = {}
_BIT_REVERSE = bytes(int("{:08b}".format(i)[::-1], 2) for i in range(256))
_GREYSCALE = bytes(i & 0x30 for i in range(256))


def _decode_pattern_row(lo, hi):
    key = (lo << 8) | hi
    row = _PATTERN_ROW_CACHE.get(key)
    if row is None:
        row = bytes(((lo >> bit) & 1) | (((hi >> bit) & 1) << 1)
                    for bit in range(7, -1, -1))
        _PATTERN_ROW_CACHE[key] = row       # at most 65536 entries
    return row


class PPU:
    def __init__(self, mapper):
        self.mapper = mapper
        self.cpu = None
        self.chr = mapper.chr
        self.chr_map = mapper.chr_map
        self.chr_is_ram = mapper.chr_is_ram
        self.vram = bytearray(4096)
        self.palette = bytearray(32)
        self.oam = bytearray(256)
        self.framebuffer = bytearray(256 * 240)
        self.frame = 0
        self.ctrl = 0
        self.mask = 0
        self.status = 0
        self.oam_addr = 0
        self.v = 0
        self.t = 0
        self.x = 0
        self.w = 0
        self.buffer = 0
        self.latch = 0
        self._a12 = 0
        self.odd_frame = False
        # Cycle-stamped flag events (absolute CPU cycles, see run_frame)
        self.line_cycle = 0.0
        self.hit_at = None
        self.vbl_at = float("inf")
        self._vbl_fired = False
        self.emphasis = 0
        # caches
        self._luts = None
        self._pal_dirty = True
        self._seg_cache = {}
        self._line_cache = {}
        self._gen = 0
        self._stamp = 0
        self._row_gen = [0] * 128
        self._spr_dirty = True
        self._spr_lines = [[] for _ in range(240)]
        self._spr_over = bytearray(240)
        for i, v in enumerate((0x09,0x01,0x00,0x01,0x00,0x02,0x02,0x0D,
                               0x08,0x10,0x08,0x24,0x00,0x00,0x04,0x2C,
                               0x09,0x01,0x34,0x03,0x00,0x04,0x00,0x14,
                               0x00,0x00,0x00,0x05,0x00,0x00,0x00,0x00)):
            self.palette[i] = v

    def reset(self):
        self.ctrl = 0
        self.mask = 0
        self.w = 0
        self.x = 0
        self.t = 0
        self.buffer = 0
        self.odd_frame = False
        self.status = 0
        self.hit_at = None
        self.vbl_at = float("inf")
        self._vbl_fired = False
        self._spr_dirty = True
        self._a12 = 0
        self.oam_addr = 0
        self.v = 0
        self.latch = 0
        self._spr_over[:] = bytes(240)
        self._line_cache.clear()

    # -- VRAM --------------------------------------------------------------
    def _nt_addr(self, addr):
        a = addr & 0x0FFF
        return (_MIRROR_PAGES[self.mapper.mirroring][a >> 10] << 10) | (a & 0x3FF)

    def _touch(self, pa):
        self._stamp += 1
        s = self._stamp
        page = (pa >> 10) << 5
        off = pa & 0x3FF
        rg = self._row_gen
        rg[page | (off >> 5)] = s
        if off >= 0x3C0:
            r0 = ((off - 0x3C0) >> 3) << 2
            for r in range(r0, min(r0 + 4, 32)):
                rg[page | r] = s

    def _bump_global(self):
        self._stamp += 1
        self._gen = self._stamp

    def ppu_read(self, addr):
        addr &= 0x3FFF
        if addr < 0x2000:
            return self.chr[self.chr_map[addr >> 10] + (addr & 0x3FF)]
        if addr < 0x3F00:
            return self.vram[self._nt_addr(addr)]
        a = addr & 0x1F
        if (a & 0x13) == 0x10:
            a &= 0x0F
        return self.palette[a]

    def ppu_write(self, addr, val):
        addr &= 0x3FFF
        if addr < 0x2000:
            if self.chr_is_ram:
                o = self.chr_map[addr >> 10] + (addr & 0x3FF)
                if self.chr[o] != val:
                    self.chr[o] = val
                    self._bump_global()
        elif addr < 0x3F00:
            pa = self._nt_addr(addr)
            if self.vram[pa] != val:
                self.vram[pa] = val
                self._touch(pa)
        else:
            a = addr & 0x1F
            if (a & 0x13) == 0x10:
                a &= 0x0F
            val &= 0x3F
            if self.palette[a] != val:
                self.palette[a] = val
                self._pal_dirty = True
                self._bump_global()

    # -- CPU-visible registers --------------------------------------------
    def reg_read(self, reg):
        if reg == 2:
            # LDA/BIT $2002 sample the bus on their 4th cycle (+3).
            now = self.cpu.cycles
            if self.hit_at is not None and now >= self.hit_at:
                self.status |= 0x40
                self.hit_at = None
            if not self._vbl_fired and now >= self.vbl_at:
                self._fire_vbl()
            v = (self.status & 0xE0) | (self.latch & 0x1F)
            self.status &= 0x7F
            self.w = 0
            return v
        if reg == 4:
            return self.oam[self.oam_addr]
        if reg == 7:
            addr = self.v & 0x3FFF
            if addr < 0x3F00:
                ret = self.buffer
                self.buffer = self.ppu_read(addr)
            else:
                ret = self.ppu_read(addr) | (self.latch & 0xC0)
                self.buffer = self.ppu_read(addr - 0x1000)
            self.v = (self.v + (32 if (self.ctrl & 4) else 1)) & 0x7FFF
            self._bus_addr(self.v)
            return ret
        return self.latch

    def reg_write(self, reg, val):
        self.latch = val
        if reg == 0:
            old = self.ctrl
            self.ctrl = val
            self.t = (self.t & 0x73FF) | ((val & 3) << 10)
            if (val ^ old) & 0x20:
                self._spr_dirty = True
            # Enabling NMI while VBlank is already set fires immediately.
            if (val & 0x80) and not (old & 0x80) and (self.status & 0x80):
                self.cpu.nmi_pending = True
                self.cpu.nmi_skip = True
        elif reg == 1:
            self.mask = val
        elif reg == 3:
            self.oam_addr = val
        elif reg == 4:
            self.oam[self.oam_addr] = val
            self.oam_addr = (self.oam_addr + 1) & 0xFF
            self._spr_dirty = True
        elif reg == 5:
            if self.w == 0:
                self.t = (self.t & 0x7FE0) | (val >> 3)
                self.x = val & 7
                self.w = 1
            else:
                self.t = (self.t & 0x0C1F) | ((val & 7) << 12) | ((val & 0xF8) << 2)
                self.w = 0
        elif reg == 6:
            if self.w == 0:
                self.t = (self.t & 0x00FF) | ((val & 0x3F) << 8)
                self.w = 1
            else:
                self.t = (self.t & 0x7F00) | val
                self.v = self.t
                self.w = 0
                self._bus_addr(self.v)
        elif reg == 7:
            self.ppu_write(self.v & 0x3FFF, val)
            self.v = (self.v + (32 if (self.ctrl & 4) else 1)) & 0x7FFF
            self._bus_addr(self.v)

    def _bus_addr(self, addr):
        """Track PPU A12 for CPU-driven address changes (rendering off)."""
        a12 = addr & 0x1000
        if a12 and not self._a12 and not (self.mask & 0x18):
            self.mapper.a12_rise()
        self._a12 = a12

    def oam_dma_write(self, data):
        a = self.oam_addr
        if a == 0:
            self.oam[:] = data
        else:
            oam = self.oam
            for b in data:
                oam[a] = b
                a = (a + 1) & 0xFF
        self._spr_dirty = True

    # -- sprite evaluation (done for all 240 lines at once) ---------------
    def _build_sprite_lines(self):
        lines = self._spr_lines
        for lst in lines:
            lst.clear()
        over = self._spr_over
        over[:] = bytes(240)
        h = 16 if (self.ctrl & 0x20) else 8
        oam = self.oam
        # Evaluation on line L selects sprites with Y <= L < Y+h and they are
        # drawn on L+1 (OAM Y = screen Y - 1).  Overflow belongs to L itself,
        # so Y=239 still overflows even though it is never drawn.
        counts = bytearray(240)
        for i in range(0, 256, 4):
            y = oam[i]
            if y == 255:
                # Do not special-case this sprite onto visible scanline 0.
                continue
            if y >= 240:
                continue
            for ev in range(y, min(y + h, 240)):
                n = counts[ev]
                if n < 8:
                    counts[ev] = n + 1
                    if ev < 239:
                        lines[ev + 1].append(i)
                else:
                    over[ev] = 1
        self._spr_dirty = False

    # -- background ------------------------------------------------------
    def _refresh_luts(self):
        pal = self.palette
        luts = []
        for p in range(4):
            lut = bytearray(256)
            lut[0] = pal[0]
            lut[1] = pal[p * 4 + 1]
            lut[2] = pal[p * 4 + 2]
            lut[3] = pal[p * 4 + 3]
            luts.append(bytes(lut))
        self._luts = luts
        self._seg_cache.clear()
        self._line_cache.clear()
        self._pal_dirty = False

    def _bg_line(self):
        """Return (colour_row, pixel_row) of 264 px for the current v."""
        v = self.v
        fine_y = (v >> 12) & 7
        cy = (v >> 5) & 31
        cx0 = v & 31
        nt = (v >> 10) & 3
        pages = _MIRROR_PAGES[self.mapper.mirroring]
        pl = pages[nt]
        pr = pages[nt ^ 1]
        bank = 4 if (self.ctrl & 0x10) else 0
        cm = self.chr_map
        rg = self._row_gen
        key = (self._gen, self._stamp if self._pal_dirty else 0, bank, fine_y, cy, cx0, pl, pr,
               rg[(pl << 5) | cy], rg[(pr << 5) | cy],
               cm[bank], cm[bank + 1], cm[bank + 2], cm[bank + 3])
        hit = self._line_cache.get(key)
        if hit is not None:
            return hit
        if self._pal_dirty:
            self._refresh_luts()
        luts = self._luts
        seg_cache = self._seg_cache
        vram = self.vram
        chr_ = self.chr
        pt = bank << 10
        row_off = cy << 5
        at_off = 0x3C0 + ((cy >> 2) << 3)
        at_sy = (cy & 2) << 1
        base = pl << 10
        cx = cx0
        cols = []
        pix = []
        for _ in range(33):
            tile = vram[base + row_off + cx]
            pal = (vram[base + at_off + (cx >> 2)] >> (at_sy | (cx & 2))) & 3
            a = pt + (tile << 4) + fine_y
            o = cm[a >> 10] + (a & 0x3FF)
            lo = chr_[o]
            hi = chr_[o + 8]
            k = (lo << 10) | (hi << 2) | pal
            seg = seg_cache.get(k)
            if seg is None:
                raw = _decode_pattern_row(lo, hi)
                seg = (raw.translate(luts[pal]), raw)
                seg_cache[k] = seg
            cols.append(seg[0])
            pix.append(seg[1])
            cx += 1
            if cx == 32:
                cx = 0
                base = pr << 10
        res = (b"".join(cols), b"".join(pix))
        lc = self._line_cache
        if len(lc) > 8192:
            lc.clear()
        lc[key] = res
        return res

    # -- scanline rendering ------------------------------------------------
    def _render_line(self, line):
        mask = self.mask
        base = line << 8
        fb = self.framebuffer
        if not (mask & 0x18):
            v = self.v & 0x3FFF
            if v >= 0x3F00:                   # palette-hack backdrop
                a = v & 0x1F
                if (a & 0x13) == 0x10:
                    a &= 0x0F
                c = self.palette[a]
            else:
                c = self.palette[0]
            fb[base:base + 256] = bytes((c,)) * 256
            return
        bg_on = mask & 0x08
        sp_on = mask & 0x10
        fx = self.x
        if bg_on:
            cols, pix = self._bg_line()
            row = bytearray(cols[fx:fx + 256])
            if not (mask & 0x02):
                row[0:8] = bytes((self.palette[0],)) * 8
        else:
            row = bytearray((self.palette[0],)) * 256
            pix = None
        if self._spr_dirty:
            self._build_sprite_lines()
        if self._spr_over[line]:             # evaluation runs if bg OR sprites on
            self.status |= 0x20
        if sp_on:
            sl = self._spr_lines[line]
            if sl:
                if pix is not None:
                    bgp = bytearray(pix[fx:fx + 256])
                    if not (mask & 0x02):
                        bgp[0:8] = bytes(8)
                else:
                    bgp = bytes(256)
                self._draw_sprites(line, sl, row, bgp, bool(bg_on))
        if mask & 0x01:
            row = row.translate(_GREYSCALE)
        fb[base:base + 256] = row

    def _draw_sprites(self, line, sl, row, bgp, bg_on):
        oam = self.oam
        chr_ = self.chr
        cm = self.chr_map
        pal = self.palette
        tall = self.ctrl & 0x20
        h = 16 if tall else 8
        sp_pt = 0x1000 if (self.ctrl & 0x08) else 0
        left = 0 if (self.mask & 0x04) else 8
        claimed = bytearray(256)
        status = self.status
        # A stale hit timestamp must not survive into a later scanline.
        if self.hit_at is not None and self.hit_at < self.line_cycle:
            self.status |= 0x40
            self.hit_at = None
        for i in sl:
            r = (line - oam[i] - 1) & 0xFF
            tile = oam[i + 1]
            attr = oam[i + 2]
            sx = oam[i + 3]
            if attr & 0x80:
                r = h - 1 - r
            if tall:
                a = ((tile & 1) << 12) + (((tile & 0xFE) + (r >> 3)) << 4) + (r & 7)
            else:
                a = sp_pt + (tile << 4) + r
            o = cm[a >> 10] + (a & 0x3FF)
            lo = chr_[o]
            hi = chr_[o + 8]
            if not (lo | hi):
                continue
            if attr & 0x40:
                lo = _BIT_REVERSE[lo]
                hi = _BIT_REVERSE[hi]
            pattern = _decode_pattern_row(lo, hi)
            pbase = 0x10 | ((attr & 3) << 2)
            behind = attr & 0x20
            zero = (i == 0) and bg_on and not (status & 0x40) and self.hit_at is None
            for px in range(8):
                p = pattern[px]
                if not p:
                    continue
                x = sx + px
                if x > 255:
                    break
                if x < left or claimed[x]:
                    continue
                claimed[x] = 1
                b = bgp[x]
                # Dot 255 cannot produce a sprite-zero hit; never schedule it.
                if zero and b and x != 255:
                    # Flag rises a couple of dots after the pixel; $2002
                    # reads compare against this CPU-cycle timestamp.
                    self.hit_at = self.line_cycle + (x + 2) / 3.0
                    zero = False
                if behind and b:
                    continue
                row[x] = pal[pbase | p]
        # Sprite-zero hit is committed by the timestamped PPU event.
        # Do not overwrite a hit raised by a prior scanline.

    # -- per-line timing hooks (called by NES.run_frame) -------------------
    def _fire_vbl(self):
        self._vbl_fired = True
        self.status |= 0x80
        if self.ctrl & 0x80:
            self.cpu.nmi_pending = True

    def begin_line(self, line):
        if self.hit_at is not None:          # commit last line's hit
            self.status |= 0x40
            self.hit_at = None
        if line < 240:
            self._render_line(line)
        elif line == 241:
            if not self._vbl_fired:
                self._fire_vbl()
        elif line == 261:
            self.status &= 0x1F

    def dot256(self, line):
        """Dots 256-260: Y increment, horizontal copy, MMC3 A12 clock."""
        if (line < 240 or line == 261) and (self.mask & 0x18):
            v = self.v
            if (v & 0x7000) != 0x7000:
                v += 0x1000
            else:
                v &= 0x0FFF
                y = (v >> 5) & 31
                if y == 29:
                    y = 0
                    v ^= 0x0800
                elif y == 31:
                    y = 0
                else:
                    y += 1
                v = (v & 0x7C1F) | (y << 5)
            v = (v & 0x7BE0) | (self.t & 0x041F)
            if line == 261:
                v = (v & 0x041F) | (self.t & 0x7BE0)
            self.v = v
        if line == 239:
            self.emphasis = (self.mask >> 5) & 7

    def _a12_late(self):
        """BG at $1000 with 8x8 sprites at $0000: A12 rises at the dot-324
        BG prefetch instead of the dot-260 sprite fetch."""
        return (self.ctrl & 0x38) in (0x00, 0x10)

    def dot260(self, line):
        if (line < 240 or line == 261) and (self.mask & 0x18) and not self._a12_late():
            self.mapper.scanline_tick()

    def dot324(self, line):
        if (line < 240 or line == 261) and (self.mask & 0x18) and self._a12_late() and (self.ctrl & 0x10):
            self.mapper.scanline_tick()

    def end_frame(self):
        self._vbl_fired = False
        self.frame += 1
        self.odd_frame = not self.odd_frame


# =============================================================================
#  APU (2A03)
# =============================================================================

_LENGTH_TABLE = (
    10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14,
    12, 16, 24, 18, 48, 20, 96, 22, 192, 24, 72, 26, 16, 28, 32, 30,
)
_NOISE_PERIOD_NTSC = (
    4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068,
)
_DMC_RATE_NTSC = (
    428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54,
)
_DUTY_SEQ = (
    (0, 1, 0, 0, 0, 0, 0, 0),
    (0, 1, 1, 0, 0, 0, 0, 0),
    (0, 1, 1, 1, 1, 0, 0, 0),
    (1, 0, 0, 1, 1, 1, 1, 1),
)
# Frame sequencer in CPU cycles (nesdev "APU Frame Counter": 3728.5 APU
# cycles = 7457 CPU cycles per quarter frame).  0.1.x used the APU-cycle
# numbers against a CPU-cycle counter, so envelopes/lengths ran at 2x.
# (cycle, quarter, half, irq).  The 4-step IRQ flag is asserted on three
# consecutive cycles 29828-29830; a $4015 read on 29829 is re-set on 29830.
_FRAME_STEPS_4 = ((7457, True, False, False), (14913, True, True, False),
                  (22371, True, False, False), (29828, False, False, True),
                  (29829, True, True, True), (29830, False, False, True))
_FRAME_PERIOD_4 = 29830
_FRAME_STEPS_5 = ((7457, True, False, False), (14913, True, True, False),
                  (22371, True, False, False), (29829, False, False, False),
                  (37281, True, True, False))
_FRAME_PERIOD_5 = 37282
# nesdev mixer lookup tables
_PULSE_MIX = tuple(0.0 if n == 0 else 95.52 / (8128.0 / n + 100.0)
                   for n in range(31))
_TND_MIX = tuple(0.0 if n == 0 else 163.67 / (24329.0 / n + 100.0)
                 for n in range(203))


class _Envelope:
    __slots__ = ()

    def clock_envelope(self):
        if self.envelope_start:
            self.envelope_start = False
            self.envelope_counter = 15
            self.envelope_divider = self.volume
        elif self.envelope_divider:
            self.envelope_divider -= 1
        else:
            self.envelope_divider = self.volume
            if self.envelope_counter:
                self.envelope_counter -= 1
            elif self.envelope_loop:
                self.envelope_counter = 15

    def clock_length(self):
        if not self.length_halt and self.length_counter:
            self.length_counter -= 1


class _Pulse(_Envelope):
    __slots__ = (
        "enabled", "channel", "timer", "timer_period", "duty", "duty_step",
        "length_counter", "length_halt", "volume", "constant_volume",
        "envelope_loop", "envelope_start", "envelope_divider", "envelope_counter",
        "sweep_enabled", "sweep_period", "sweep_negate", "sweep_shift",
        "sweep_reload", "sweep_divider", "target_period",
    )

    def __init__(self, channel):
        self.channel = channel
        self.reset()

    def reset(self):
        self.enabled = False
        self.timer = 0
        self.timer_period = 0
        self.duty = 0
        self.duty_step = 0
        self.length_counter = 0
        self.length_halt = False
        self.volume = 0
        self.constant_volume = False
        self.envelope_loop = False
        self.envelope_start = False
        self.envelope_divider = 0
        self.envelope_counter = 0
        self.sweep_enabled = False
        self.sweep_period = 0
        self.sweep_negate = False
        self.sweep_shift = 0
        self.sweep_reload = False
        self.sweep_divider = 0
        self.target_period = 0

    def write_reg0(self, val):
        self.duty = (val >> 6) & 3
        self.length_halt = self.envelope_loop = bool(val & 0x20)
        self.constant_volume = bool(val & 0x10)
        self.volume = val & 0x0F

    def write_reg1(self, val):
        self.sweep_enabled = bool(val & 0x80)
        self.sweep_period = (val >> 4) & 7
        self.sweep_negate = bool(val & 0x08)
        self.sweep_shift = val & 7
        self.sweep_reload = True
        self._update_target()

    def write_reg2(self, val):
        self.timer_period = (self.timer_period & 0x700) | val
        self._update_target()

    def write_reg3(self, val):
        self.timer_period = (self.timer_period & 0xFF) | ((val & 7) << 8)
        if self.enabled:
            self.length_counter = _LENGTH_TABLE[val >> 3]
        self.duty_step = 0
        self.envelope_start = True
        self._update_target()

    def _update_target(self):
        change = self.timer_period >> self.sweep_shift
        if self.sweep_negate:
            change = -change - 1 if self.channel == 0 else -change
        self.target_period = self.timer_period + change

    def clock_timer(self, cycles):
        if cycles <= self.timer:
            self.timer -= cycles
            return
        period = (self.timer_period + 1) << 1
        remaining = cycles - self.timer - 1
        self.timer = period - 1 - (remaining % period)
        self.duty_step = (self.duty_step + 1 + remaining // period) & 7

    def clock_sweep(self):
        if (self.sweep_divider == 0 and self.sweep_enabled and self.sweep_shift
                and self.timer_period >= 8 and 0 <= self.target_period <= 0x7FF):
            self.timer_period = self.target_period
            self._update_target()
        if self.sweep_divider == 0 or self.sweep_reload:
            self.sweep_divider = self.sweep_period
            self.sweep_reload = False
        else:
            self.sweep_divider -= 1

    def output(self):
        # Muting by target > $7FF applies even when the sweep unit is off.
        if (not self.length_counter or self.timer_period < 8
                or self.target_period > 0x7FF):
            return 0
        if not _DUTY_SEQ[self.duty][self.duty_step]:
            return 0
        return self.volume if self.constant_volume else self.envelope_counter


class _Triangle:
    __slots__ = (
        "enabled", "timer", "timer_period", "length_counter", "length_halt",
        "linear_counter", "linear_reload_value", "linear_reload_flag",
        "control_flag", "step",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.enabled = False
        self.timer = 0
        self.timer_period = 0
        self.length_counter = 0
        self.length_halt = False
        self.linear_counter = 0
        self.linear_reload_value = 0
        self.linear_reload_flag = False
        self.control_flag = False
        self.step = 0

    def write_reg0(self, val):
        self.control_flag = self.length_halt = bool(val & 0x80)
        self.linear_reload_value = val & 0x7F

    def write_reg2(self, val):
        self.timer_period = (self.timer_period & 0x700) | val

    def write_reg3(self, val):
        self.timer_period = (self.timer_period & 0xFF) | ((val & 7) << 8)
        if self.enabled:
            self.length_counter = _LENGTH_TABLE[val >> 3]
        self.linear_reload_flag = True

    def clock_timer(self, cycles):
        if cycles <= self.timer:
            self.timer -= cycles
            return
        period = self.timer_period + 1
        remaining = cycles - self.timer - 1
        self.timer = period - 1 - (remaining % period)
        if self.timer_period >= 2:
            self.step = (self.step + 1 + remaining // period) & 31

    def clock_linear(self):
        if self.linear_reload_flag:
            self.linear_counter = self.linear_reload_value
        elif self.linear_counter:
            self.linear_counter -= 1
        if not self.control_flag:
            self.linear_reload_flag = False

    def clock_length(self):
        if not self.length_halt and self.length_counter:
            self.length_counter -= 1

    def output(self):
        if self.timer_period < 2:
            return 7                          # ultrasonic: hold mid-level
        s = self.step
        return 15 - s if s < 16 else s - 16


class _Noise(_Envelope):
    __slots__ = (
        "enabled", "timer", "timer_period", "mode", "shift", "length_counter",
        "length_halt", "volume", "constant_volume", "envelope_loop",
        "envelope_start", "envelope_divider", "envelope_counter",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.enabled = False
        self.timer = 0
        self.timer_period = _NOISE_PERIOD_NTSC[0]
        self.mode = 0
        self.shift = 1
        self.length_counter = 0
        self.length_halt = False
        self.volume = 0
        self.constant_volume = False
        self.envelope_loop = False
        self.envelope_start = False
        self.envelope_divider = 0
        self.envelope_counter = 0

    def write_reg0(self, val):
        self.length_halt = self.envelope_loop = bool(val & 0x20)
        self.constant_volume = bool(val & 0x10)
        self.volume = val & 0x0F

    def write_reg2(self, val):
        self.mode = (val >> 7) & 1
        self.timer_period = _NOISE_PERIOD_NTSC[val & 0x0F]

    def write_reg3(self, val):
        if self.enabled:
            self.length_counter = _LENGTH_TABLE[val >> 3]
        self.envelope_start = True

    def clock_timer(self, cycles):
        if cycles <= self.timer:
            self.timer -= cycles
            return
        period = self.timer_period
        remaining = cycles - self.timer - 1
        steps = 1 + remaining // period
        self.timer = period - 1 - (remaining % period)
        tap = 6 if self.mode else 1
        s = self.shift
        for _ in range(steps):
            s = (s >> 1) | (((s ^ (s >> tap)) & 1) << 14)
        self.shift = s

    def output(self):
        if not self.length_counter or (self.shift & 1):
            return 0
        return self.volume if self.constant_volume else self.envelope_counter


class _DMC:
    __slots__ = (
        "enabled", "irq_enabled", "loop", "timer", "rate_period",
        "output_level", "sample_addr", "sample_length", "current_addr",
        "bytes_remaining", "shift", "bits_remaining", "silence", "irq_pending",
        "sample_buffer", "sample_buffer_empty", "read", "cpu",
    )

    def __init__(self):
        self.read = None
        self.cpu = None
        self.reset()

    def reset(self):
        self.enabled = False
        self.irq_enabled = False
        self.loop = False
        self.timer = 0
        self.rate_period = _DMC_RATE_NTSC[0]
        self.output_level = 0
        self.sample_addr = 0xC000
        self.sample_length = 1
        self.current_addr = 0xC000
        self.bytes_remaining = 0
        self.shift = 0
        self.bits_remaining = 8
        self.silence = True
        self.irq_pending = False
        self.sample_buffer = 0
        self.sample_buffer_empty = True

    def write_reg0(self, val):
        self.irq_enabled = bool(val & 0x80)
        self.loop = bool(val & 0x40)
        self.rate_period = _DMC_RATE_NTSC[val & 0x0F]
        if not self.irq_enabled:
            self.irq_pending = False

    def write_reg1(self, val):
        self.output_level = val & 0x7F

    def write_reg2(self, val):
        self.sample_addr = 0xC000 | (val << 6)

    def write_reg3(self, val):
        self.sample_length = (val << 4) | 1

    def start(self):
        if not self.bytes_remaining:
            self.current_addr = self.sample_addr
            self.bytes_remaining = self.sample_length
        self._fill()

    def active(self):
        return self.bytes_remaining or not self.silence or not self.sample_buffer_empty

    def clock_timer(self, cycles):
        if cycles <= self.timer:
            self.timer -= cycles
            return
        period = self.rate_period
        remaining = cycles - self.timer - 1
        steps = 1 + remaining // period
        self.timer = period - 1 - (remaining % period)
        for _ in range(steps):
            if not self.silence:
                if self.shift & 1:
                    if self.output_level <= 125:
                        self.output_level += 2
                elif self.output_level >= 2:
                    self.output_level -= 2
                self.shift >>= 1
            self.bits_remaining -= 1
            if self.bits_remaining <= 0:
                self.bits_remaining = 8
                if self.sample_buffer_empty:
                    self.silence = True
                else:
                    self.silence = False
                    self.shift = self.sample_buffer
                    self.sample_buffer_empty = True
                    self._fill()

    def _fill(self):
        if not self.sample_buffer_empty or not self.bytes_remaining:
            return
        self.sample_buffer = self.read(self.current_addr) if self.read else 0
        if self.cpu is not None:
            self.cpu.cycles += 4              # DMA steals the CPU
        self.sample_buffer_empty = False
        self.current_addr = 0x8000 if self.current_addr == 0xFFFF else self.current_addr + 1
        self.bytes_remaining -= 1
        if not self.bytes_remaining:
            if self.loop:
                self.current_addr = self.sample_addr
                self.bytes_remaining = self.sample_length
            elif self.irq_enabled:
                self.irq_pending = True


class APU:
    """Lazily synchronised: the CPU never calls into the APU per instruction.
    sync() catches up to cpu.cycles on register access and at every
    scanline boundary."""

    GAIN = 2.0 * 32767.0

    def __init__(self):
        self.cpu = None
        self.audio = None
        self.pulse1 = _Pulse(0)
        self.pulse2 = _Pulse(1)
        self.triangle = _Triangle()
        self.noise = _Noise()
        self.dmc = _DMC()
        self.frame_mode = 0
        self.frame_irq_inhibit = False
        self.frame_irq = False
        self.frame_cycle = 0
        self.frame_step = 0
        self._next_step = _FRAME_STEPS_4[0][0]
        self.synced = 0
        self.irq_at = float("inf")
        self._to_next = int(CYCLES_PER_SAMPLE)
        self._frac = CYCLES_PER_SAMPLE - self._to_next
        self._hp_in = 0.0
        self._hp_out = 0.0
        self._lp = 0.0
        self._hp_r = math.exp(-2.0 * math.pi * 90.0 / SAMPLE_RATE)
        self._lp_a = 1.0 - math.exp(-2.0 * math.pi * 14000.0 / SAMPLE_RATE)
        self.pcm = array("h")

    def attach(self, cpu, read):
        self.cpu = cpu
        self.dmc.cpu = cpu
        self.dmc.read = read
        self.synced = cpu.cycles

    @property
    def emitting(self):
        a = self.audio
        return AUDIO_ENABLED and a is not None and a.active

    # -- sync ---------------------------------------------------------------
    def sync(self, ahead=0):
        """Catch up to cpu.cycles + ahead.  Register accesses pass ahead=3:
        LDA/STA abs touch the bus on their 4th cycle, while cpu.cycles still
        holds the instruction's first cycle."""
        now = self.cpu.cycles + ahead
        d = now - self.synced
        if d <= 0:
            return
        self.synced = now
        if self.emitting:
            clock = self._clock
            emit = self._emit
            cps = CYCLES_PER_SAMPLE
            while d:
                k = self._to_next
                if d < k:
                    clock(d)
                    self._to_next = k - d
                    break
                clock(k)
                d -= k
                emit()
                f = self._frac + cps
                k = int(f)
                self._frac = f - k
                self._to_next = k
        else:
            self._clock(d)
        # Cycles stolen by DMC DMA during this catch-up land after `now`
        # and are picked up by the next sync().
        self._irq_update()
        self._schedule_irq()

    def _schedule_irq(self):
        """Absolute CPU cycle at which the 4-step frame IRQ will assert, so
        NES.run_frame can stop the CPU exactly there instead of at the next
        scanline boundary."""
        if self.frame_mode or self.frame_irq_inhibit or self.frame_irq:
            self.irq_at = float("inf")
        else:
            for cyc, _q, _h, irq in _FRAME_STEPS_4[self.frame_step:]:
                if irq:
                    break
            self.irq_at = self.synced + max(1, cyc - self.frame_cycle)

    def _clock(self, k):
        p = self.pulse1
        if p.length_counter:
            p.clock_timer(k)
        p = self.pulse2
        if p.length_counter:
            p.clock_timer(k)
        t = self.triangle
        if t.length_counter and t.linear_counter:
            t.clock_timer(k)
        n = self.noise
        if n.length_counter:
            n.clock_timer(k)
        d = self.dmc
        if d.enabled or not d.silence:
            d.clock_timer(k)
        fc = self.frame_cycle + k
        self.frame_cycle = fc
        if fc >= self._next_step:
            self._frame_seq()

    def _frame_seq(self):
        if self.frame_mode:
            steps, period = _FRAME_STEPS_5, _FRAME_PERIOD_5
        else:
            steps, period = _FRAME_STEPS_4, _FRAME_PERIOD_4
        # Re-evaluate the current step after each event, including wraps.
        while self.frame_cycle >= steps[self.frame_step][0]:
            _, quarter, half, irq = steps[self.frame_step]
            if quarter:
                self._quarter()
            if half:
                self._half()
            if irq and not self.frame_irq_inhibit:
                self.frame_irq = True
            self.frame_step += 1
            if self.frame_step == len(steps):
                self.frame_step = 0
                self.frame_cycle -= period
        self._next_step = steps[self.frame_step][0]

    def _quarter(self):
        self.pulse1.clock_envelope()
        self.pulse2.clock_envelope()
        self.triangle.clock_linear()
        self.noise.clock_envelope()

    def _half(self):
        self.pulse1.clock_length()
        self.pulse2.clock_length()
        self.triangle.clock_length()
        self.noise.clock_length()
        self.pulse1.clock_sweep()
        self.pulse2.clock_sweep()

    def _irq_update(self):
        cpu = self.cpu
        cpu.irq = (cpu.irq & 1) | (2 if self.frame_irq else 0) | \
                  (4 if self.dmc.irq_pending else 0)

    def _emit(self):
        mixed = (_PULSE_MIX[self.pulse1.output() + self.pulse2.output()]
                 + _TND_MIX[3 * self.triangle.output() + 2 * self.noise.output()
                            + self.dmc.output_level])
        hp = mixed - self._hp_in + self._hp_r * self._hp_out
        self._hp_in = mixed
        self._hp_out = hp
        lp = self._lp + self._lp_a * (hp - self._lp)
        self._lp = lp
        s = int(lp * self.GAIN)
        if s > 32767:
            s = 32767
        elif s < -32768:
            s = -32768
        self.pcm.append(s)

    def take_pcm(self):
        pcm = self.pcm
        self.pcm = array("h")
        return pcm

    # -- registers ------------------------------------------------------------
    def write(self, addr, val):
        self.sync(3)
        if addr == 0x4017:
            self.frame_mode = 1 if (val & 0x80) else 0
            self.frame_irq_inhibit = bool(val & 0x40)
            if self.frame_irq_inhibit:
                self.frame_irq = False
            # Sequencer restarts 3-4 CPU cycles after the write.
            self.frame_cycle = -(4 if (self.synced & 1) else 3)
            self.frame_step = 0
            self._next_step = (_FRAME_STEPS_5 if self.frame_mode else _FRAME_STEPS_4)[0][0]
            if self.frame_mode:
                self._quarter()
                self._half()
            self._irq_update()
            self._schedule_irq()
            return
        r = addr - 0x4000
        if r == 0x15:
            self.pulse1.enabled = bool(val & 0x01)
            self.pulse2.enabled = bool(val & 0x02)
            self.triangle.enabled = bool(val & 0x04)
            self.noise.enabled = bool(val & 0x08)
            self.dmc.enabled = bool(val & 0x10)
            if not self.pulse1.enabled:
                self.pulse1.length_counter = 0
            if not self.pulse2.enabled:
                self.pulse2.length_counter = 0
            if not self.triangle.enabled:
                self.triangle.length_counter = 0
            if not self.noise.enabled:
                self.noise.length_counter = 0
            self.dmc.irq_pending = False
            if val & 0x10:
                self.dmc.start()
            else:
                self.dmc.bytes_remaining = 0
            self._irq_update()
            return
        if r < 4:
            (self.pulse1.write_reg0, self.pulse1.write_reg1,
             self.pulse1.write_reg2, self.pulse1.write_reg3)[r](val)
        elif r < 8:
            (self.pulse2.write_reg0, self.pulse2.write_reg1,
             self.pulse2.write_reg2, self.pulse2.write_reg3)[r - 4](val)
        elif r == 0x08:
            self.triangle.write_reg0(val)
        elif r == 0x0A:
            self.triangle.write_reg2(val)
        elif r == 0x0B:
            self.triangle.write_reg3(val)
        elif r == 0x0C:
            self.noise.write_reg0(val)
        elif r == 0x0E:
            self.noise.write_reg2(val)
        elif r == 0x0F:
            self.noise.write_reg3(val)
        elif 0x10 <= r <= 0x13:
            (self.dmc.write_reg0, self.dmc.write_reg1,
             self.dmc.write_reg2, self.dmc.write_reg3)[r - 0x10](val)
            self._irq_update()

    def read_status(self):
        self.sync(3)
        s = 0
        if self.pulse1.length_counter:
            s |= 0x01
        if self.pulse2.length_counter:
            s |= 0x02
        if self.triangle.length_counter:
            s |= 0x04
        if self.noise.length_counter:
            s |= 0x08
        if self.dmc.bytes_remaining:
            s |= 0x10
        if self.frame_irq:
            s |= 0x40
        if self.dmc.irq_pending:
            s |= 0x80
        self.frame_irq = False
        self._irq_update()
        self._schedule_irq()
        return s

    def reset(self):
        self.write(0x4015, 0)
        self.frame_irq = False
        self.dmc.irq_pending = False
        self._irq_update()


# =============================================================================
#  Native audio output (WinMM / CoreAudio / ALSA via ctypes)
# =============================================================================

class AudioStream:
    """Mono s16 PCM.  The emulator pushes one block per frame."""

    _CHUNK = 512
    _POOL = 6
    TARGET_FILL = 1600            # software queue target (samples, ~33 ms)
    _MAX_FILL = SAMPLE_RATE // 5  # hard cap (200 ms) before trimming
    WHDR_DONE = 0x00000001
    WHDR_PREPARED = 0x00000002

    def __init__(self, rate=SAMPLE_RATE):
        self.rate = rate
        self._buf = bytearray()
        self._last = b"\x00\x00"
        self._lock = threading.Lock()
        self._alive = False
        self._running = False
        self._backend = "none"
        self._play_thread = None
        self._winmm = None
        self._hwo = None
        self._pool = []
        self._pool_free = set()
        self._coreaudio = None
        self._audio_queue = None
        self._coreaudio_callback = None
        self._core_buffers = []
        self._alsa = None
        self._alsa_pcm = None
        self._alsa_error_handler = None

    @property
    def active(self):
        return self._running and self._backend != "none"

    def status(self):
        if not AUDIO_ENABLED:
            return "audio off"
        if self.active:
            names = {"winmm": "WinMM", "coreaudio": "CoreAudio", "alsa": "ALSA"}
            return f"audio on ({names.get(self._backend, self._backend)}, {self.rate} Hz)"
        return "audio unavailable"

    def buffered(self):
        return len(self._buf) >> 1

    def start(self):
        if not AUDIO_ENABLED or self.active:
            return
        self._alive = True
        if sys.platform == "win32":
            self._open_winmm()
        elif sys.platform == "darwin":
            self._open_coreaudio()
        elif sys.platform.startswith("linux"):
            self._open_alsa()
        if self._backend == "none":
            self._alive = False
            return
        self._running = True
        if self._backend in ("winmm", "alsa"):
            self._play_thread = threading.Thread(
                target=self._play_loop, name="luigines-audio", daemon=True)
            self._play_thread.start()

    def push_pcm(self, pcm):
        if not self.active or not pcm:
            return
        data = pcm.tobytes() if isinstance(pcm, array) else bytes(pcm)
        if sys.byteorder != "little":
            a = array("h", data)
            a.byteswap()
            data = a.tobytes()
        with self._lock:
            self._buf += data
            excess = len(self._buf) - self._MAX_FILL * 2
            if excess > 0:
                drop = len(self._buf) - self.TARGET_FILL * 2
                del self._buf[:drop & ~1]

    def _take_pcm(self, frames):
        n = frames * 2
        with self._lock:
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
        if len(chunk) < n:
            # Underrun: hold the last sample instead of dropping to zero
            # so a late frame produces a flat spot, not a click.
            pad = chunk[-2:] if len(chunk) >= 2 else self._last
            chunk += pad * ((n - len(chunk)) >> 1)
        self._last = chunk[-2:]
        return chunk

    def stop(self):
        self._alive = False
        self._running = False
        try:
            if self._backend == "winmm" and self._winmm is not None and self._hwo:
                self._winmm.waveOutReset(self._hwo)
            elif self._backend == "alsa" and self._alsa is not None and self._alsa_pcm:
                self._alsa.snd_pcm_drop(self._alsa_pcm)
            elif (self._backend == "coreaudio" and self._coreaudio is not None
                  and self._audio_queue):
                self._coreaudio.AudioQueueStop(self._audio_queue, True)
        except OSError:
            pass
        if (self._play_thread is not None
                and self._play_thread is not threading.current_thread()):
            self._play_thread.join(timeout=0.5)
        if self._backend == "winmm":
            self._close_winmm()
        elif self._backend == "coreaudio":
            self._close_coreaudio()
        elif self._backend == "alsa":
            self._close_alsa()
        with self._lock:
            self._buf.clear()
        self._backend = "none"

    # -- Windows: waveOut -------------------------------------------------
    def _open_winmm(self):
        try:
            winmm = ctypes.windll.winmm
        except (AttributeError, OSError):
            return

        class WAVEFORMATEX(ctypes.Structure):
            _fields_ = [
                ("wFormatTag", ctypes.c_uint16), ("nChannels", ctypes.c_uint16),
                ("nSamplesPerSec", ctypes.c_uint32),
                ("nAvgBytesPerSec", ctypes.c_uint32),
                ("nBlockAlign", ctypes.c_uint16), ("wBitsPerSample", ctypes.c_uint16),
                ("cbSize", ctypes.c_uint16),
            ]

        class WAVEHDR(ctypes.Structure):
            _fields_ = [
                ("lpData", ctypes.c_void_p), ("dwBufferLength", ctypes.c_uint32),
                ("dwBytesRecorded", ctypes.c_uint32), ("dwUser", ctypes.c_size_t),
                ("dwFlags", ctypes.c_uint32), ("dwLoops", ctypes.c_uint32),
                ("lpNext", ctypes.c_void_p), ("reserved", ctypes.c_size_t),
            ]

        winmm.waveOutOpen.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
                                      ctypes.POINTER(WAVEFORMATEX), ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_uint]
        winmm.waveOutPrepareHeader.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
        winmm.waveOutWrite.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
        winmm.waveOutUnprepareHeader.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
        winmm.waveOutReset.argtypes = [ctypes.c_void_p]
        winmm.waveOutClose.argtypes = [ctypes.c_void_p]
        fmt = WAVEFORMATEX(1, 1, self.rate, self.rate * 2, 2, 16, 0)
        handle = ctypes.c_void_p()
        if winmm.waveOutOpen(ctypes.byref(handle), 0xFFFFFFFF, ctypes.byref(fmt),
                             None, None, 0) != 0:
            return
        self._winmm = winmm
        self._hwo = handle
        self._pool = []
        self._pool_free = set(range(self._POOL))
        for _ in range(self._POOL):
            buf = ctypes.create_string_buffer(self._CHUNK * 2)
            hdr = WAVEHDR(ctypes.cast(buf, ctypes.c_void_p), self._CHUNK * 2,
                          0, 0, 0, 0, None, 0)
            self._pool.append((buf, hdr))
        self._backend = "winmm"

    def _release_winmm_buffer(self, idx):
        if idx in self._pool_free:
            return
        _, hdr = self._pool[idx]
        if hdr.dwFlags & self.WHDR_PREPARED:
            self._winmm.waveOutUnprepareHeader(
                self._hwo, ctypes.byref(hdr), ctypes.sizeof(hdr))
        hdr.dwFlags = 0
        self._pool_free.add(idx)

    def _winmm_loop(self):
        while self._alive:
            for idx, (_, hdr) in enumerate(self._pool):
                if idx not in self._pool_free and hdr.dwFlags & self.WHDR_DONE:
                    self._release_winmm_buffer(idx)
            in_flight = self._POOL - len(self._pool_free)
            queued = self.buffered()
            # Feed a full chunk when one is queued; if the device is down to
            # its last buffer, send a padded partial chunk so it never stops.
            if not self._pool_free or not (
                    queued >= self._CHUNK or (queued and in_flight <= 1)):
                time.sleep(0.001)
                continue
            idx = min(self._pool_free)
            self._pool_free.remove(idx)
            buf, hdr = self._pool[idx]
            pcm = self._take_pcm(self._CHUNK)
            ctypes.memmove(buf, pcm, len(pcm))
            hdr.dwBufferLength = len(pcm)
            hdr.dwFlags = 0
            if self._winmm.waveOutPrepareHeader(
                    self._hwo, ctypes.byref(hdr), ctypes.sizeof(hdr)) != 0:
                self._pool_free.add(idx)
                continue
            if self._winmm.waveOutWrite(
                    self._hwo, ctypes.byref(hdr), ctypes.sizeof(hdr)) != 0:
                self._release_winmm_buffer(idx)

    def _close_winmm(self):
        if self._winmm is None or not self._hwo:
            return
        self._winmm.waveOutReset(self._hwo)
        for idx in range(len(self._pool)):
            self._release_winmm_buffer(idx)
        self._winmm.waveOutClose(self._hwo)
        self._hwo = None
        self._winmm = None
        self._pool.clear()

    # -- macOS: AudioQueue ------------------------------------------------
    def _open_coreaudio(self):
        try:
            audio = ctypes.CDLL(
                "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox")
        except OSError:
            return

        class AudioStreamBasicDescription(ctypes.Structure):
            _fields_ = [
                ("mSampleRate", ctypes.c_double), ("mFormatID", ctypes.c_uint32),
                ("mFormatFlags", ctypes.c_uint32), ("mBytesPerPacket", ctypes.c_uint32),
                ("mFramesPerPacket", ctypes.c_uint32), ("mBytesPerFrame", ctypes.c_uint32),
                ("mChannelsPerFrame", ctypes.c_uint32), ("mBitsPerChannel", ctypes.c_uint32),
                ("mReserved", ctypes.c_uint32),
            ]

        class AudioQueueBuffer(ctypes.Structure):
            _fields_ = [
                ("mAudioDataBytesCapacity", ctypes.c_uint32),
                ("mAudioData", ctypes.c_void_p),
                ("mAudioDataByteSize", ctypes.c_uint32),
                ("mUserData", ctypes.c_void_p),
                ("mPacketDescriptionCapacity", ctypes.c_uint32),
                ("mPacketDescriptions", ctypes.c_void_p),
            ]

        buffer_ref = ctypes.POINTER(AudioQueueBuffer)
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, buffer_ref)
        audio.AudioQueueNewOutput.argtypes = [
            ctypes.POINTER(AudioStreamBasicDescription), callback_type,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        audio.AudioQueueAllocateBuffer.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(buffer_ref)]
        audio.AudioQueueEnqueueBuffer.argtypes = [
            ctypes.c_void_p, buffer_ref, ctypes.c_uint32, ctypes.c_void_p]
        audio.AudioQueueStart.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        audio.AudioQueueStop.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
        audio.AudioQueueDispose.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]

        def fill_buffer(_user, queue, buffer):
            if not self._alive:
                return
            pcm = self._take_pcm(self._CHUNK)
            ctypes.memmove(buffer.contents.mAudioData, pcm, len(pcm))
            buffer.contents.mAudioDataByteSize = len(pcm)
            audio.AudioQueueEnqueueBuffer(queue, buffer, 0, None)

        callback = callback_type(fill_buffer)
        # 'lpcm', signed integer | packed, mono 16-bit little-endian.
        fmt = AudioStreamBasicDescription(
            float(self.rate), 0x6C70636D, 0x0C, 2, 1, 2, 1, 16, 0)
        queue = ctypes.c_void_p()
        if audio.AudioQueueNewOutput(
                ctypes.byref(fmt), callback, None, None, None, 0,
                ctypes.byref(queue)) != 0:
            return
        self._coreaudio = audio
        self._audio_queue = queue
        self._coreaudio_callback = callback
        self._core_buffers = []
        for _ in range(self._POOL):
            buffer = buffer_ref()
            if audio.AudioQueueAllocateBuffer(
                    queue, self._CHUNK * 2, ctypes.byref(buffer)) != 0:
                self._close_coreaudio()
                return
            self._core_buffers.append(buffer)
            fill_buffer(None, queue, buffer)
        if audio.AudioQueueStart(queue, None) != 0:
            self._close_coreaudio()
            return
        self._backend = "coreaudio"

    def _close_coreaudio(self):
        if self._coreaudio is not None and self._audio_queue:
            self._coreaudio.AudioQueueStop(self._audio_queue, True)
            self._coreaudio.AudioQueueDispose(self._audio_queue, True)
        self._audio_queue = None
        self._coreaudio = None
        self._coreaudio_callback = None
        self._core_buffers.clear()

    # -- Linux: ALSA ------------------------------------------------------
    def _open_alsa(self):
        name = ctypes.util.find_library("asound") or "libasound.so.2"
        try:
            alsa = ctypes.CDLL(name)
        except OSError:
            return
        error_type = ctypes.CFUNCTYPE(
            None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_char_p)
        self._alsa_error_handler = error_type(lambda *_args: None)
        try:
            alsa.snd_lib_error_set_handler(self._alsa_error_handler)
        except AttributeError:
            pass
        alsa.snd_pcm_open.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                      ctypes.c_int, ctypes.c_int]
        alsa.snd_pcm_set_params.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_uint,
                                            ctypes.c_uint, ctypes.c_int,
                                            ctypes.c_uint]
        alsa.snd_pcm_writei.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        alsa.snd_pcm_writei.restype = ctypes.c_long
        alsa.snd_pcm_recover.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        alsa.snd_pcm_drop.argtypes = [ctypes.c_void_p]
        alsa.snd_pcm_close.argtypes = [ctypes.c_void_p]
        pcm = ctypes.c_void_p()
        if alsa.snd_pcm_open(ctypes.byref(pcm), b"default", 0, 0) < 0:
            return
        # S16_LE=2, RW_INTERLEAVED=3, mono, resample on, 60 ms device latency.
        if alsa.snd_pcm_set_params(pcm, 2, 3, 1, self.rate, 1, 60000) < 0:
            alsa.snd_pcm_close(pcm)
            return
        self._alsa = alsa
        self._alsa_pcm = pcm
        self._backend = "alsa"

    def _alsa_loop(self):
        while self._alive:
            if self.buffered() < self._CHUNK:
                time.sleep(0.001)
                continue
            pcm = self._take_pcm(self._CHUNK)
            data = ctypes.create_string_buffer(pcm, len(pcm))
            result = self._alsa.snd_pcm_writei(
                self._alsa_pcm, ctypes.cast(data, ctypes.c_void_p), self._CHUNK)
            if result < 0:
                self._alsa.snd_pcm_recover(self._alsa_pcm, int(result), 1)

    def _close_alsa(self):
        if self._alsa is not None and self._alsa_pcm:
            self._alsa.snd_pcm_drop(self._alsa_pcm)
            self._alsa.snd_pcm_close(self._alsa_pcm)
        self._alsa_pcm = None
        self._alsa = None
        self._alsa_error_handler = None

    def _play_loop(self):
        if self._backend == "winmm":
            self._winmm_loop()
        elif self._backend == "alsa":
            self._alsa_loop()


# =============================================================================
#  Controller
# =============================================================================

class Controller:
    BTN_A      = 0x01
    BTN_B      = 0x02
    BTN_SELECT = 0x04
    BTN_START  = 0x08
    BTN_UP     = 0x10
    BTN_DOWN   = 0x20
    BTN_LEFT   = 0x40
    BTN_RIGHT  = 0x80

    def __init__(self):
        self.state = 0
        self.shift = 0
        self.strobe = False

    def set_strobe(self, val):
        was_high = self.strobe
        self.strobe = bool(val & 1)
        if self.strobe or was_high:
            self.shift = self.state

    def read(self):
        if self.strobe:
            return self.state & 1
        v = self.shift & 1
        self.shift = (self.shift >> 1) | 0x80
        return v


# =============================================================================
#  CPU bus  (read/write are plain closures: no method dispatch on hot paths)
# =============================================================================

class Bus:
    def __init__(self, mapper, ppu, apu, pad1, pad2):
        self.ram = ram = bytearray(2048)
        self.mapper = mapper
        self.ppu = ppu
        self.apu = apu
        self.pad1 = pad1
        self.pad2 = pad2
        self.cpu = None
        prg = mapper.prg
        prg_map = mapper.prg_map
        ppu_rr = ppu.reg_read
        ppu_rw = ppu.reg_write
        m_read = mapper.cpu_read
        m_write = mapper.cpu_write
        bus = self

        def read(addr):
            if addr >= 0x8000:
                return prg[prg_map[(addr >> 13) & 3] + (addr & 0x1FFF)]
            if addr < 0x2000:
                return ram[addr & 0x7FF]
            if addr < 0x4000:
                return ppu_rr(addr & 7)
            if addr < 0x4020:
                if addr == 0x4016:
                    return 0x40 | pad1.read()
                if addr == 0x4017:
                    return 0x40 | pad2.read()
                if addr == 0x4015:
                    return apu.read_status()
                return 0x40
            return m_read(addr)

        def write(addr, val):
            if addr < 0x2000:
                ram[addr & 0x7FF] = val
            elif addr < 0x4000:
                ppu_rw(addr & 7, val)
            elif addr < 0x4020:
                if addr == 0x4014:
                    page = val << 8
                    if page < 0x2000:
                        a = page & 0x7FF
                        data = bytes(ram[a:a + 256])
                    else:
                        data = bytes(read(page | i) for i in range(256))
                    ppu.oam_dma_write(data)
                    c = bus.cpu
                    c.cycles += 513 + (c.cycles & 1)
                elif addr == 0x4016:
                    pad1.set_strobe(val)
                    pad2.set_strobe(val)
                else:
                    apu.write(addr, val)
            else:
                m_write(addr, val)

        self.read = read
        self.write = write


# =============================================================================
#  6502 CPU (2A03 core: no decimal mode)
#
#  Each of the 256 opcodes is compiled from the spec below into its own
#  straight-line closure: addressing mode inlined, flags from a ZN lookup,
#  zero-page and stack going straight to the RAM bytearray.
# =============================================================================

# mnemonic  mode:opcode:cycles ...      (cycles exclude page-cross penalty)
_CPU_SPEC = """
ADC imm:69:2 zp:65:3 zpx:75:4 abs:6D:4 absx:7D:4 absy:79:4 indx:61:6 indy:71:5
SBC imm:E9:2 zp:E5:3 zpx:F5:4 abs:ED:4 absx:FD:4 absy:F9:4 indx:E1:6 indy:F1:5 imm:EB:2
AND imm:29:2 zp:25:3 zpx:35:4 abs:2D:4 absx:3D:4 absy:39:4 indx:21:6 indy:31:5
ORA imm:09:2 zp:05:3 zpx:15:4 abs:0D:4 absx:1D:4 absy:19:4 indx:01:6 indy:11:5
EOR imm:49:2 zp:45:3 zpx:55:4 abs:4D:4 absx:5D:4 absy:59:4 indx:41:6 indy:51:5
CMP imm:C9:2 zp:C5:3 zpx:D5:4 abs:CD:4 absx:DD:4 absy:D9:4 indx:C1:6 indy:D1:5
CPX imm:E0:2 zp:E4:3 abs:EC:4
CPY imm:C0:2 zp:C4:3 abs:CC:4
BIT zp:24:3 abs:2C:4
LDA imm:A9:2 zp:A5:3 zpx:B5:4 abs:AD:4 absx:BD:4 absy:B9:4 indx:A1:6 indy:B1:5
LDX imm:A2:2 zp:A6:3 zpy:B6:4 abs:AE:4 absy:BE:4
LDY imm:A0:2 zp:A4:3 zpx:B4:4 abs:AC:4 absx:BC:4
LAX zp:A7:3 zpy:B7:4 abs:AF:4 absy:BF:4 indx:A3:6 indy:B3:5
LXA imm:AB:2
LAS absy:BB:4
STA zp:85:3 zpx:95:4 abs:8D:4 absx:9D:5 absy:99:5 indx:81:6 indy:91:6
STX zp:86:3 zpy:96:4 abs:8E:4
STY zp:84:3 zpx:94:4 abs:8C:4
SAX zp:87:3 zpy:97:4 abs:8F:4 indx:83:6
SHY absx:9C:5
SHX absy:9E:5
AHX absy:9F:5 indy:93:6
TAS absy:9B:5
ASL acc:0A:2 zp:06:5 zpx:16:6 abs:0E:6 absx:1E:7
LSR acc:4A:2 zp:46:5 zpx:56:6 abs:4E:6 absx:5E:7
ROL acc:2A:2 zp:26:5 zpx:36:6 abs:2E:6 absx:3E:7
ROR acc:6A:2 zp:66:5 zpx:76:6 abs:6E:6 absx:7E:7
INC zp:E6:5 zpx:F6:6 abs:EE:6 absx:FE:7
DEC zp:C6:5 zpx:D6:6 abs:CE:6 absx:DE:7
SLO zp:07:5 zpx:17:6 abs:0F:6 absx:1F:7 absy:1B:7 indx:03:8 indy:13:8
RLA zp:27:5 zpx:37:6 abs:2F:6 absx:3F:7 absy:3B:7 indx:23:8 indy:33:8
SRE zp:47:5 zpx:57:6 abs:4F:6 absx:5F:7 absy:5B:7 indx:43:8 indy:53:8
RRA zp:67:5 zpx:77:6 abs:6F:6 absx:7F:7 absy:7B:7 indx:63:8 indy:73:8
DCP zp:C7:5 zpx:D7:6 abs:CF:6 absx:DF:7 absy:DB:7 indx:C3:8 indy:D3:8
ISB zp:E7:5 zpx:F7:6 abs:EF:6 absx:FF:7 absy:FB:7 indx:E3:8 indy:F3:8
ANC imm:0B:2 imm:2B:2
ALR imm:4B:2
ARR imm:6B:2
AXS imm:CB:2
XAA imm:8B:2
NOP imp:EA:2 imp:1A:2 imp:3A:2 imp:5A:2 imp:7A:2 imp:DA:2 imp:FA:2 imm:80:2 imm:82:2 imm:89:2 imm:C2:2 imm:E2:2 zp:04:3 zp:44:3 zp:64:3 zpx:14:4 zpx:34:4 zpx:54:4 zpx:74:4 zpx:D4:4 zpx:F4:4 abs:0C:4 absx:1C:4 absx:3C:4 absx:5C:4 absx:7C:4 absx:DC:4 absx:FC:4
BPL rel:10:2
BMI rel:30:2
BVC rel:50:2
BVS rel:70:2
BCC rel:90:2
BCS rel:B0:2
BNE rel:D0:2
BEQ rel:F0:2
JMP abs:4C:3 ind:6C:5
JSR abs:20:6
RTS imp:60:6
RTI imp:40:6
BRK imp:00:7
PHA imp:48:3
PHP imp:08:3
PLA imp:68:4
PLP imp:28:4
CLC imp:18:2
SEC imp:38:2
CLI imp:58:2
SEI imp:78:2
CLV imp:B8:2
CLD imp:D8:2
SED imp:F8:2
TAX imp:AA:2
TAY imp:A8:2
TXA imp:8A:2
TYA imp:98:2
TSX imp:BA:2
TXS imp:9A:2
INX imp:E8:2
INY imp:C8:2
DEX imp:CA:2
DEY imp:88:2
JAM imp:02:2 imp:12:2 imp:22:2 imp:32:2 imp:42:2 imp:52:2 imp:62:2 imp:72:2 imp:92:2 imp:B2:2 imp:D2:2 imp:F2:2
"""

_ZP_MODES = ("zp", "zpx", "zpy")

_MODE_SRC = {
    "imm":  ["pc = c.pc", "v = read(pc)", "c.pc = (pc + 1) & 0xFFFF"],
    "zp":   ["pc = c.pc", "addr = read(pc)", "c.pc = (pc + 1) & 0xFFFF"],
    "zpx":  ["pc = c.pc", "addr = (read(pc) + c.x) & 0xFF", "c.pc = (pc + 1) & 0xFFFF"],
    "zpy":  ["pc = c.pc", "addr = (read(pc) + c.y) & 0xFF", "c.pc = (pc + 1) & 0xFFFF"],
    "abs":  ["pc = c.pc", "addr = read(pc) | (read((pc + 1) & 0xFFFF) << 8)",
             "c.pc = (pc + 2) & 0xFFFF"],
    "absx": ["pc = c.pc", "base = read(pc) | (read((pc + 1) & 0xFFFF) << 8)",
             "c.pc = (pc + 2) & 0xFFFF", "addr = (base + c.x) & 0xFFFF"],
    "absy": ["pc = c.pc", "base = read(pc) | (read((pc + 1) & 0xFFFF) << 8)",
             "c.pc = (pc + 2) & 0xFFFF", "addr = (base + c.y) & 0xFFFF"],
    "indx": ["pc = c.pc", "z = (read(pc) + c.x) & 0xFF", "c.pc = (pc + 1) & 0xFFFF",
             "addr = ram[z] | (ram[(z + 1) & 0xFF] << 8)"],
    "indy": ["pc = c.pc", "z = read(pc)", "c.pc = (pc + 1) & 0xFFFF",
             "base = ram[z] | (ram[(z + 1) & 0xFF] << 8)",
             "addr = (base + c.y) & 0xFFFF"],
}

# Bodies operating on operand value v (read group)
_READ_OPS = {
    "LDA": ["c.a = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "LDX": ["c.x = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "LDY": ["c.y = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "LAX": ["c.a = c.x = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "LXA": ["c.a = c.x = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "LAS": ["v &= c.sp", "c.a = c.x = c.sp = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "ORA": ["v |= c.a", "c.a = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "AND": ["v &= c.a", "c.a = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "EOR": ["v ^= c.a", "c.a = v", "c.p = (c.p & 0x7D) | ZN[v]"],
    "ADC": ["a = c.a", "p = c.p", "r = a + v + (p & 1)", "c.a = r & 0xFF",
            "c.p = (p & 0x3C) | (r >> 8) | (((a ^ r) & (v ^ r) & 0x80) >> 1) | ZN[r & 0xFF]"],
    "SBC": ["v ^= 0xFF", "a = c.a", "p = c.p", "r = a + v + (p & 1)", "c.a = r & 0xFF",
            "c.p = (p & 0x3C) | (r >> 8) | (((a ^ r) & (v ^ r) & 0x80) >> 1) | ZN[r & 0xFF]"],
    "CMP": ["r = c.a - v", "c.p = (c.p & 0x7C) | (r >= 0) | ZN[r & 0xFF]"],
    "CPX": ["r = c.x - v", "c.p = (c.p & 0x7C) | (r >= 0) | ZN[r & 0xFF]"],
    "CPY": ["r = c.y - v", "c.p = (c.p & 0x7C) | (r >= 0) | ZN[r & 0xFF]"],
    "BIT": ["c.p = (c.p & 0x3D) | (v & 0xC0) | (0 if (c.a & v) else 2)"],
    "NOP": [],
    "ANC": ["v &= c.a", "c.a = v", "c.p = (c.p & 0x7C) | (v >> 7) | ZN[v]"],
    "ALR": ["v &= c.a", "r = v >> 1", "c.a = r", "c.p = (c.p & 0x7C) | (v & 1) | ZN[r]"],
    "ARR": ["r = ((c.a & v) >> 1) | ((c.p & 1) << 7)", "c.a = r",
            "c.p = (c.p & 0x3C) | ((r >> 6) & 1) | ((((r >> 6) ^ (r >> 5)) & 1) << 6) | ZN[r]"],
    "AXS": ["r = (c.a & c.x) - v", "c.x = r & 0xFF",
            "c.p = (c.p & 0x7C) | (r >= 0) | ZN[r & 0xFF]"],
    "XAA": ["v &= c.x", "c.a = v", "c.p = (c.p & 0x7D) | ZN[v]"],
}

# Read-modify-write kernels: m -> r (flags set here)
_RMW_OPS = {
    "ASL": ["r = (m << 1) & 0xFF", "c.p = (c.p & 0x7C) | (m >> 7) | ZN[r]"],
    "LSR": ["r = m >> 1", "c.p = (c.p & 0x7C) | (m & 1) | ZN[r]"],
    "ROL": ["r = ((m << 1) | (c.p & 1)) & 0xFF", "c.p = (c.p & 0x7C) | (m >> 7) | ZN[r]"],
    "ROR": ["r = (m >> 1) | ((c.p & 1) << 7)", "c.p = (c.p & 0x7C) | (m & 1) | ZN[r]"],
    "INC": ["r = (m + 1) & 0xFF", "c.p = (c.p & 0x7D) | ZN[r]"],
    "DEC": ["r = (m - 1) & 0xFF", "c.p = (c.p & 0x7D) | ZN[r]"],
}
# Unofficial RMW combos: (rmw kernel, read op applied to v = r)
_COMBO_OPS = {
    "SLO": ("ASL", "ORA"), "RLA": ("ROL", "AND"), "SRE": ("LSR", "EOR"),
    "RRA": ("ROR", "ADC"), "DCP": ("DEC", "CMP"), "ISB": ("INC", "SBC"),
}
_PAGE_PENALTY = {"ADC", "SBC", "AND", "ORA", "EOR", "CMP", "LDA", "LDX", "LDY",
                 "LAX", "LAS", "NOP"}

_STORE_VAL = {"STA": "c.a", "STX": "c.x", "STY": "c.y", "SAX": "c.a & c.x"}

_BRANCH_COND = {
    "BPL": "not (c.p & 0x80)", "BMI": "c.p & 0x80",
    "BVC": "not (c.p & 0x40)", "BVS": "c.p & 0x40",
    "BCC": "not (c.p & 0x01)", "BCS": "c.p & 0x01",
    "BNE": "not (c.p & 0x02)", "BEQ": "c.p & 0x02",
}

_PUSH = ["ram[0x100 | sp] = {v}", "sp = (sp - 1) & 0xFF"]

_IMPLIED = {
    "CLC": ["c.p &= 0xFE"], "SEC": ["c.p |= 0x01"],
    # CLI/SEI/PLP change I after the IRQ poll, so the *old* I decides
    # whether an IRQ is taken right after them (blargg cli_latency).
    "CLI": ["c.ipoll_i = c.p & 0x04", "c.ipoll_cycle = c.cycles + 2", "c.p &= 0xFB"],
    "SEI": ["c.ipoll_i = c.p & 0x04", "c.ipoll_cycle = c.cycles + 2", "c.p |= 0x04"],
    "CLV": ["c.p &= 0xBF"], "CLD": ["c.p &= 0xF7"], "SED": ["c.p |= 0x08"],
    "TAX": ["v = c.x = c.a", "c.p = (c.p & 0x7D) | ZN[v]"],
    "TAY": ["v = c.y = c.a", "c.p = (c.p & 0x7D) | ZN[v]"],
    "TXA": ["v = c.a = c.x", "c.p = (c.p & 0x7D) | ZN[v]"],
    "TYA": ["v = c.a = c.y", "c.p = (c.p & 0x7D) | ZN[v]"],
    "TSX": ["v = c.x = c.sp", "c.p = (c.p & 0x7D) | ZN[v]"],
    "TXS": ["c.sp = c.x"],
    "INX": ["v = c.x = (c.x + 1) & 0xFF", "c.p = (c.p & 0x7D) | ZN[v]"],
    "INY": ["v = c.y = (c.y + 1) & 0xFF", "c.p = (c.p & 0x7D) | ZN[v]"],
    "DEX": ["v = c.x = (c.x - 1) & 0xFF", "c.p = (c.p & 0x7D) | ZN[v]"],
    "DEY": ["v = c.y = (c.y - 1) & 0xFF", "c.p = (c.p & 0x7D) | ZN[v]"],
    "NOP": [],
    "PHA": ["sp = c.sp"] + [s.format(v="c.a") for s in _PUSH] + ["c.sp = sp"],
    "PHP": ["sp = c.sp"] + [s.format(v="c.p | 0x30") for s in _PUSH] + ["c.sp = sp"],
    "PLA": ["sp = (c.sp + 1) & 0xFF", "c.sp = sp", "v = c.a = ram[0x100 | sp]",
            "c.p = (c.p & 0x7D) | ZN[v]"],
    "PLP": ["c.ipoll_i = c.p & 0x04", "c.ipoll_cycle = c.cycles + 4",
            "sp = (c.sp + 1) & 0xFF", "c.sp = sp",
            "c.p = (ram[0x100 | sp] & 0xCF) | 0x20"],
    "RTS": ["sp = c.sp", "lo = ram[0x100 | ((sp + 1) & 0xFF)]",
            "sp = (sp + 2) & 0xFF", "hi = ram[0x100 | sp]", "c.sp = sp",
            "c.pc = (((hi << 8) | lo) + 1) & 0xFFFF"],
    "RTI": ["sp = (c.sp + 1) & 0xFF", "c.p = (ram[0x100 | sp] & 0xCF) | 0x20",
            "sp = (sp + 1) & 0xFF", "lo = ram[0x100 | sp]",
            "sp = (sp + 1) & 0xFF", "hi = ram[0x100 | sp]", "c.sp = sp",
            "c.pc = (hi << 8) | lo"],
    "BRK": ["pc = (c.pc + 1) & 0xFFFF", "sp = c.sp"]
           + [s.format(v="pc >> 8") for s in _PUSH]
           + [s.format(v="pc & 0xFF") for s in _PUSH]
           + [s.format(v="c.p | 0x30") for s in _PUSH]
           + ["c.sp = sp", "c.p |= 0x04", "c.pc = read(0xFFFE) | (read(0xFFFF) << 8)"],
    "JAM": ["c.jammed = True", "c.pc = (c.pc - 1) & 0xFFFF",
            "return 2"],
}


def _gen_op(mn, mode, cy):
    """Return the source lines (body) of one opcode handler."""
    L = []
    if mode == "rel":
        L += ["pc = c.pc",
              f"if {_BRANCH_COND[mn]}:",
              "    off = read(pc)",
              "    pc = (pc + 1) & 0xFFFF",
              "    t = (pc + ((off ^ 0x80) - 0x80)) & 0xFFFF",
              "    c.pc = t",
              "    return 4 if (pc ^ t) & 0xFF00 else 3",
              "c.pc = (pc + 1) & 0xFFFF",
              "return 2"]
        return L
    if mn == "JMP":
        L += ["pc = c.pc", "ptr = read(pc) | (read((pc + 1) & 0xFFFF) << 8)"]
        if mode == "abs":
            L += ["c.pc = ptr"]
        else:   # 6502 page-wrap bug on the indirect vector
            L += ["c.pc = read(ptr) | (read((ptr & 0xFF00) | ((ptr + 1) & 0xFF)) << 8)"]
        return L + [f"return {cy}"]
    if mn == "JSR":
        L += ["pc = c.pc", "lo = read(pc)", "hi = read((pc + 1) & 0xFFFF)",
              "ret = (pc + 1) & 0xFFFF", "sp = c.sp"]
        L += [s.format(v="ret >> 8") for s in _PUSH]
        L += [s.format(v="ret & 0xFF") for s in _PUSH]
        L += ["c.sp = sp", "c.pc = lo | (hi << 8)", f"return {cy}"]
        return L
    if mode == "imp" or mode == "acc":
        if mode == "acc":
            L += ["m = c.a"] + _RMW_OPS[mn] + ["c.a = r"]
        else:
            L += _IMPLIED[mn]
        if not L or not L[-1].startswith("return"):
            L.append(f"return {cy}")
        return L

    L += _MODE_SRC[mode]
    zp = mode in _ZP_MODES
    load = "ram[addr]" if zp else "read(addr)"

    def store(val):
        return f"ram[addr] = {val}" if zp else f"write(addr, {val})"

    if mn in _READ_OPS:
        if mode != "imm":
            L.append(f"v = {load}")
        L += _READ_OPS[mn]
        if mn in _PAGE_PENALTY and mode in ("absx", "absy", "indy"):
            L.append(f"return {cy + 1} if (base ^ addr) & 0xFF00 else {cy}")
        else:
            L.append(f"return {cy}")
        return L
    if mn in _STORE_VAL:
        L += [store(_STORE_VAL[mn]), f"return {cy}"]
        return L
    if mn in ("SHY", "SHX", "AHX", "TAS"):
        src = {"SHY": "c.y", "SHX": "c.x", "AHX": "c.a & c.x"}.get(mn)
        if mn == "TAS":
            L.append("c.sp = c.a & c.x")
            src = "c.sp"
        L += ["h = ((base >> 8) + 1) & 0xFF",
              f"val = {src} & h",
              "if (base ^ addr) & 0xFF00:",
              "    addr = (val << 8) | (addr & 0xFF)",
              "write(addr, val)", f"return {cy}"]
        return L
    kernel, after = (mn, None) if mn in _RMW_OPS else _COMBO_OPS[mn]
    L.append(f"m = {load}")
    L += _RMW_OPS[kernel]
    if not zp:
        # Real 6502 RMW: dummy write of the unmodified value, then the result.
        L.append("if addr >= 0x2000:")
        L.append("    write(addr, m)")
    L.append(store("r"))
    if after:
        L.append("v = r")
        L += _READ_OPS[after]
    L.append(f"return {cy}")
    return L


def _build_cpu_factory():
    entries = {}
    for line in _CPU_SPEC.strip().splitlines():
        parts = line.split()
        mn = parts[0]
        for tok in parts[1:]:
            mode, code, cy = tok.split(":")
            code = int(code, 16)
            if code in entries:
                raise RuntimeError(f"opcode ${code:02X} defined twice")
            entries[code] = (mn, mode, int(cy))
    if len(entries) != 256:
        missing = sorted(set(range(256)) - set(entries))
        raise RuntimeError("missing opcodes: " + " ".join(f"{m:02X}" for m in missing))
    src = ["def _factory(c, read, write, ram, ZN):"]
    for code in range(256):
        mn, mode, cy = entries[code]
        src.append(f"    def op_{code:02X}():  # {mn} {mode}")
        src += ["        " + s for s in _gen_op(mn, mode, cy)]
    src.append("    return [" + ", ".join(f"op_{i:02X}" for i in range(256)) + "]")
    ns = {}
    exec(compile("\n".join(src), "<luigines-cpu>", "exec"), ns)
    names = tuple(entries[i][0] for i in range(256))
    return ns["_factory"], names, "\n".join(src)


_CPU_FACTORY, OPCODE_NAMES, _CPU_SOURCE = _build_cpu_factory()
_ZN = tuple(0x02 if v == 0 else (v & 0x80) for v in range(256))


class CPU:
    __slots__ = ("a", "x", "y", "sp", "p", "pc", "cycles", "nmi_pending",
                 "irq", "jammed", "target", "bus", "ipoll_i", "ipoll_cycle", "nmi_skip", "_ops", "_read", "_ram",
                 "_prg", "_prg_map")

    FLAG_C = 0x01
    FLAG_Z = 0x02
    FLAG_I = 0x04
    FLAG_D = 0x08
    FLAG_B = 0x10
    FLAG_U = 0x20
    FLAG_V = 0x40
    FLAG_N = 0x80

    def __init__(self, bus):
        self.bus = bus
        bus.cpu = self
        self.a = self.x = self.y = 0
        self.sp = 0xFD
        self.p = 0x24
        self.pc = 0
        self.cycles = 0
        self.nmi_pending = False
        self.irq = 0              # bit0 mapper, bit1 APU frame, bit2 DMC
        self.jammed = False
        self.target = 0
        self.ipoll_i = 0
        self.ipoll_cycle = -1
        self.nmi_skip = False
        self._read = bus.read
        self._ram = bus.ram
        self._prg = bus.mapper.prg
        self._prg_map = bus.mapper.prg_map
        self._ops = _CPU_FACTORY(self, bus.read, bus.write, bus.ram, _ZN)

    def reset(self):
        read = self._read
        self.pc = read(0xFFFC) | (read(0xFFFD) << 8)
        self.sp = (self.sp - 3) & 0xFF
        self.p |= self.FLAG_I
        self.jammed = False
        self.nmi_pending = False
        self.nmi_skip = False
        self.irq = 0
        self.ipoll_cycle = -1
        self.ipoll_i = 0
        self.cycles += 7

    def power_on(self):
        self.a = self.x = self.y = 0
        self.sp = 0xFD
        self.p = 0x24
        read = self._read
        self.pc = read(0xFFFC) | (read(0xFFFD) << 8)
        self.cycles = 7
        self.jammed = False
        self.nmi_pending = False
        self.nmi_skip = False
        self.irq = 0
        self.ipoll_cycle = -1
        self.ipoll_i = 0

    def nmi(self):
        self.nmi_pending = True

    def _interrupt(self, vector):
        ram = self._ram
        pc = self.pc
        sp = self.sp
        ram[0x100 | sp] = pc >> 8
        sp = (sp - 1) & 0xFF
        ram[0x100 | sp] = pc & 0xFF
        sp = (sp - 1) & 0xFF
        ram[0x100 | sp] = (self.p & 0xEF) | 0x20
        self.sp = (sp - 1) & 0xFF
        self.p |= 0x04
        read = self._read
        self.pc = read(vector) | (read(vector + 1) << 8)
        self.cycles += 7

    def run(self, target):
        """Execute whole instructions until cycles >= target."""
        self.target = target
        ops = self._ops
        read = self._read
        prg = self._prg
        prg_map = self._prg_map
        while self.cycles < target:
            if self.jammed:
                # A halted CPU still advances to the caller's timing target.
                self.cycles = target
                break
            if self.nmi_pending:
                if self.nmi_skip:            # $2000 NMI-enable during VBL:
                    self.nmi_skip = False    # one more instruction first
                else:
                    self.nmi_pending = False
                    if not self.jammed:
                        self._interrupt(0xFFFA)
                        continue
            if self.irq and not self.jammed:
                if self.cycles == self.ipoll_cycle:
                    masked = self.ipoll_i
                else:
                    masked = self.p & 0x04
                if not masked:
                    self._interrupt(0xFFFE)
                    continue
            pc = self.pc
            if pc >= 0x8000:
                op = prg[prg_map[(pc >> 13) & 3] + (pc & 0x1FFF)]
            else:
                op = read(pc)
            self.pc = (pc + 1) & 0xFFFF
            # Two statements on purpose: `self.cycles += ops[op]()` would load
            # cycles *before* the call and drop the 513/514 OAM-DMA or DMC
            # steal cycles the handler adds.
            cy = ops[op]()
            self.cycles += cy

    def step(self):
        """Execute exactly one instruction (or interrupt). Returns cycles."""
        start = self.cycles
        self.run(start + 1)
        return self.cycles - start


# =============================================================================
#  NES system
# =============================================================================

class NES:
    def __init__(self, rom_path, audio=None):
        self.rom = ROM(rom_path)
        self.mapper = make_mapper(self.rom)
        self.ppu = PPU(self.mapper)
        self.apu = APU()
        self.pad1 = Controller()
        self.pad2 = Controller()
        self.bus = Bus(self.mapper, self.ppu, self.apu, self.pad1, self.pad2)
        self.cpu = CPU(self.bus)
        self.mapper.cpu = self.cpu
        self.ppu.cpu = self.cpu
        self.apu.attach(self.cpu, self.bus.read)
        self.apu.audio = audio
        self.cpu.power_on()
        self.apu.synced = self.cpu.cycles
        self._cpu_target = float(self.cpu.cycles)

    def reset(self):
        self.cpu.reset()
        self.ppu.reset()
        self.apu.reset()
        self._cpu_target = float(self.cpu.cycles)

    def _run(self, target):
        apu = self.apu
        # Instructions are atomic: IRQ edges can occur inside one instruction.
        # Sync at the actual overshot CPU cycle, never at the old deadline.
        if self.cpu.cycles < apu.irq_at < target:
            self.cpu.run(apu.irq_at)
            apu.sync()
        self.cpu.run(target)

    def run_frame(self):
        run = self._run
        ppu = self.ppu
        apu_sync = self.apu.sync
        begin = ppu.begin_line
        dot256 = ppu.dot256
        dot260 = ppu.dot260
        dot324 = ppu.dot324
        per_line = CPU_CYCLES_PER_LINE
        d256 = CPU_CYCLES_AT_DOT256
        d260 = CPU_CYCLES_AT_DOT260
        d324 = CPU_CYCLES_AT_DOT324
        t = self._cpu_target
        ppu.vbl_at = t + 241 * per_line + 1.0 / 3.0     # line 241, dot 1
        for line in range(PPU_SCANLINES):
            ppu.line_cycle = t
            begin(line)
            run(t + d256)
            dot256(line)
            run(t + d260)
            dot260(line)
            run(t + d324)
            dot324(line)
            # The skipped pre-render dot is between prefetch and line 0,
            # not an adjustment to an already completed visible scanline.
            line_end = t + per_line
            if line == 261 and ppu.odd_frame and (ppu.mask & 0x18):
                line_end -= 1.0 / 3.0
            run(line_end)
            t = line_end
        apu_sync()
        ppu.end_frame()
        self._cpu_target = max(t, float(self.cpu.cycles))
        apu = self.apu
        if apu.pcm:
            pcm = apu.take_pcm()
            if apu.audio is not None:
                apu.audio.push_pcm(pcm)


# =============================================================================
#  Tkinter GUI
# =============================================================================

SCALE = 2
SCREEN_W = 256
SCREEN_H = 240
APP_VERSION = "0.1.1"
WINDOW_TITLE = "LuigiNES 0.1.1 - AC Kondo / ACHolding"
DEVELOPER_CREDIT = "Team Flames / AC Kondo / ACHolding"
NINTENDO_CREDIT = "Nintendo game titles, characters and trademarks © Nintendo."
INDEPENDENCE_NOTICE = "LuigiNES is an independent emulator, not affiliated with Nintendo."

KEYMAP_P1 = {
    "x": Controller.BTN_A, "X": Controller.BTN_A,
    "z": Controller.BTN_B, "Z": Controller.BTN_B,
    "Return": Controller.BTN_START,
    "Shift_R": Controller.BTN_SELECT, "Shift_L": Controller.BTN_SELECT,
    "Up": Controller.BTN_UP, "Down": Controller.BTN_DOWN,
    "Left": Controller.BTN_LEFT, "Right": Controller.BTN_RIGHT,
    "w": Controller.BTN_UP, "W": Controller.BTN_UP,
    "s": Controller.BTN_DOWN, "S": Controller.BTN_DOWN,
    "a": Controller.BTN_LEFT, "A": Controller.BTN_LEFT,
    "d": Controller.BTN_RIGHT, "D": Controller.BTN_RIGHT,
}

_MAX_CATCHUP_FRAMES = 3
_SPIN_SEC = 0.002


def _win_timer_resolution(enable):
    """Tk's after() rides the Windows 15.6 ms tick unless 1 ms is requested."""
    if sys.platform != "win32":
        return
    try:
        winmm = ctypes.windll.winmm
        (winmm.timeBeginPeriod if enable else winmm.timeEndPeriod)(1)
    except (AttributeError, OSError):
        pass


class EmuApp:
    BG = "#d4d0c8"
    VIEWPORT_BG = "#000000"
    STATUS_BG = "#d4d0c8"
    STATUS_FG = "#000000"

    def __init__(self, root, rom_path=None):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg=self.BG)
        self.root.minsize(SCREEN_W * SCALE + 28, SCREEN_H * SCALE + 112)
        self.root.resizable(False, False)
        self.nes = None
        self.running = False
        self.paused = False
        self.audio = None
        self.fps = 0.0
        self._fps_frames = 0
        self._fps_draws = 0
        self._fps_t0 = time.perf_counter()
        self._frame_clock = time.perf_counter() + FRAME_SEC
        self._release_jobs = {}
        self._after_job = None
        self._closing = False
        self._rgb = bytearray(SCREEN_W * SCREEN_H * 3)
        self._ppm_header = b"P6\n256 240\n255\n"
        self._blit_mode = "ppm"
        self._build_menu()
        self._build_widgets()
        self._bind_keys()
        self.root.update_idletasks()
        _win_timer_resolution(True)
        if AUDIO_ENABLED:
            self.audio = AudioStream()
            try:
                self.audio.start()
            except (OSError, RuntimeError) as exc:
                print(f"Audio unavailable: {exc}", file=sys.stderr)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        if rom_path and os.path.isfile(rom_path):
            self._open(rom_path)
        else:
            self._show_splash()
        self._schedule_tick(1)

    # -- UI construction ---------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root, relief="raised", bd=1)
        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Open NES ROM...", command=self._open_dialog,
                           accelerator="Ctrl+O")
        m_file.add_command(label="Close ROM", command=self._close_rom)
        m_file.add_command(label="ROM Information...", command=self._rom_info)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self._quit, accelerator="Alt+F4")
        menubar.add_cascade(label="File", menu=m_file)

        m_cpu = tk.Menu(menubar, tearoff=0)
        m_cpu.add_command(label="Reset", command=self._reset, accelerator="Ctrl+R")
        m_cpu.add_command(label="Power Cycle", command=self._power_cycle)
        m_cpu.add_separator()
        m_cpu.add_command(label="Pause", command=self._toggle_pause, accelerator="P")
        menubar.add_cascade(label="Emulation", menu=m_cpu)

        m_opt = tk.Menu(menubar, tearoff=0)
        m_opt.add_command(label=f"Zoom {SCALE}x (fixed)", state="disabled")
        m_opt.add_command(label="Display... 256x240 NTSC", state="disabled")
        audio_cfg = (f"Sound... 2A03 {SAMPLE_RATE} Hz"
                     if AUDIO_ENABLED and not AUDIO_FILES else "Sound... off")
        m_opt.add_command(label=audio_cfg, state="disabled")
        m_opt.add_separator()
        m_opt.add_command(label=f"Speed... locked {TARGET_FPS:.0f} FPS",
                          state="disabled")
        m_opt.add_command(label="Input... keyboard", command=self._controls_help)
        menubar.add_cascade(label="Options", menu=m_opt)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="Controls", command=self._controls_help)
        m_help.add_separator()
        m_help.add_command(label="About LuigiNES 0.1.1 / Credits...", command=self._about)
        menubar.add_cascade(label="Help", menu=m_help)
        self.root.config(menu=menubar)

    def _build_widgets(self):
        # Classic Windows 9x / Visual Basic-era emulator presentation.
        toolbar = tk.Frame(self.root, bg=self.BG, bd=2, relief="raised")
        toolbar.pack(side="top", fill="x")
        for label, command in (("Open ROM", self._open_dialog),
                               ("Reset", self._reset),
                               ("Pause / Resume", self._toggle_pause),
                               ("ROM Info", self._rom_info)):
            tk.Button(toolbar, text=label, command=command,
                      bg=self.BG, fg="#000000", relief="raised", bd=2,
                      font=("TkDefaultFont", 9), padx=8).pack(
                          side="left", padx=2, pady=3)
        tk.Label(toolbar, text="LuigiNES 0.1.1 | Team Flames / AC Kondo / ACHolding", bg=self.BG,
                 fg="#000080", font=("TkDefaultFont", 9, "bold")
                 ).pack(side="right", padx=7)

        status_frame = tk.Frame(self.root, bg=self.STATUS_BG, bd=1, relief="sunken")
        status_frame.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Ready - No ROM loaded")
        tk.Label(status_frame, textvariable=self.status_var, anchor="w",
                 bg=self.STATUS_BG, fg=self.STATUS_FG,
                 font=("TkDefaultFont", 9)).pack(
                     side="left", fill="x", expand=True, padx=4, pady=2)
        self.status_fps_var = tk.StringVar(value="--- fps")
        tk.Label(status_frame, textvariable=self.status_fps_var, anchor="e",
                 bg=self.STATUS_BG, fg=self.STATUS_FG,
                 font=("TkDefaultFont", 9)).pack(side="right", padx=4, pady=2)

        screen_frame = tk.Frame(self.root, bg=self.BG, bd=3, relief="sunken")
        screen_frame.pack(side="top", padx=7, pady=7)
        cw, ch = SCREEN_W * SCALE, SCREEN_H * SCALE
        self.canvas = tk.Canvas(screen_frame, width=cw, height=ch,
                                bg=self.VIEWPORT_BG, highlightthickness=0, bd=0)
        self.canvas.pack()
        self._src = tk.PhotoImage(width=SCREEN_W, height=SCREEN_H)
        self.photo = tk.PhotoImage(width=cw, height=ch)
        self.canvas_img = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def _rom_info(self):
        if self.nes is None:
            messagebox.showinfo("ROM Information", "No NES ROM is currently loaded.")
            return
        h = self.nes.rom.header
        messagebox.showinfo("ROM Information",
                            f"File: {h.title}\nMapper: {h.mapper}\n"
                            f"PRG ROM: {h.prg_banks * 16} KB\n"
                            f"CHR ROM: {h.chr_banks * 8} KB\n"
                            f"Mirroring: {h.mirroring}\n"
                            f"Battery-backed: {'Yes' if h.battery else 'No'}\n"
                            f"Header: {'NES 2.0' if h.is_nes2 else 'iNES'}")

    def _bind_keys(self):
        self.root.bind("<KeyPress>", self._on_key_down)
        self.root.bind("<KeyRelease>", self._on_key_up)
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<Control-o>", lambda e: self._open_dialog())
        self.root.bind("<Control-O>", lambda e: self._open_dialog())
        self.root.bind("<Control-r>", lambda e: self._reset())
        self.root.bind("<Control-R>", lambda e: self._reset())
        self.root.bind("<KeyPress-p>", lambda e: self._toggle_pause() if not (e.state & 0x04) else None)
        self.root.bind("<Pause>", lambda e: self._toggle_pause())

    # -- input (X11 auto-repeat sends release+press pairs; debounce them) ---
    def _on_focus_out(self, _ev):
        for job in self._release_jobs.values():
            self.root.after_cancel(job)
        self._release_jobs.clear()
        if self.nes is not None:
            self.nes.pad1.state = 0

    def _on_key_down(self, ev):
        job = self._release_jobs.pop(ev.keysym, None)
        if job is not None:
            self.root.after_cancel(job)
        bit = KEYMAP_P1.get(ev.keysym)
        if bit is not None and self.nes is not None:
            self.nes.pad1.state |= bit

    def _on_key_up(self, ev):
        bit = KEYMAP_P1.get(ev.keysym)
        if bit is None:
            return
        old = self._release_jobs.pop(ev.keysym, None)
        if old is not None:
            self.root.after_cancel(old)

        def release(sym=ev.keysym, b=bit):
            self._release_jobs.pop(sym, None)
            if self.nes is not None:
                self.nes.pad1.state &= ~b & 0xFF
        self._release_jobs[ev.keysym] = self.root.after(12, release)

    # -- ROM lifecycle -------------------------------------------------------
    def _show_splash(self):
        cw, ch = SCREEN_W * SCALE, SCREEN_H * SCALE
        self.photo.blank()
        self.canvas.delete("splash")
        self.canvas.create_text(cw // 2, ch // 2 - 12, anchor="center",
                                text="LuigiNES 0.1.1", fill="#c0c0c0",
                                font=("TkDefaultFont", 16, "bold"), tags="splash")
        self.canvas.create_text(cw // 2, ch // 2 + 14, anchor="center",
                                text="Team Flames / AC Kondo / ACHolding\nFile > Open NES ROM... to begin",
                                fill="#808080", font=("TkDefaultFont", 10),
                                tags="splash")

    def _hide_splash(self):
        self.canvas.delete("splash")

    def _rom_display_name(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        return name.replace("_", " ").replace(".", " ")

    def _open_dialog(self):
        path = filedialog.askopenfilename(
            title="Open NES ROM",
            filetypes=[("NES iNES ROM", "*.nes"), ("All files", "*.*")])
        if path:
            self._open(path)

    def _open(self, path):
        if self.audio is None and AUDIO_ENABLED:
            self.audio = AudioStream()
        if self.audio is not None:
            self.audio.start()
        try:
            nes = NES(path, self.audio)
        except Exception as e:
            self.running = False
            self.paused = False
            if self.nes is not None:
                self.nes.cpu.jammed = False
                self.nes.apu.audio = None
            self.nes = None
            self.status_var.set("ROM load failed")
            messagebox.showerror("luigines", f"Failed to load ROM:\n{path}\n\n{e}")
            return
        self.nes = nes
        self.running = True
        self.paused = False
        self._fps_frames = self._fps_draws = 0
        self._fps_t0 = time.perf_counter()
        self._frame_clock = time.perf_counter() + FRAME_SEC
        self._hide_splash()
        h = nes.rom.header
        astat = self.audio.status() if self.audio else "audio off"
        self.status_var.set(f"{self._rom_display_name(path)}  |  mapper {h.mapper}"
                            f"  |  {astat}")
        self.status_fps_var.set("--- fps")

    def _close_rom(self):
        if self.nes:
            self.nes.apu.audio = None
        self.nes = None
        self.running = False
        self.status_var.set("No ROM loaded.")
        self.status_fps_var.set("--- fps")
        self._show_splash()

    def _reset(self):
        if self.nes:
            self.nes.reset()
            self.running = True

    def _power_cycle(self):
        if self.nes:
            self._open(self.nes.rom.header.filename)

    def _toggle_pause(self):
        if not self.nes:
            return
        self.paused = not self.paused
        s = self.status_var.get().replace("  [paused]", "")
        self.status_var.set(s + ("  [paused]" if self.paused else ""))
        self._frame_clock = time.perf_counter() + FRAME_SEC

    def _about(self):
        messagebox.showinfo(
            "About LuigiNES",
            f"{WINDOW_TITLE}\n"
            f"Developed by: {DEVELOPER_CREDIT}\n"
            "© AC Kondo / ACHolding 1999–2026\n"
            f"{NINTENDO_CREDIT}\n"
            f"{INDEPENDENCE_NOTICE}\n\n"
            "Classic Windows NES emulator interface inspired by MarioNES 1.5.\n"
            "MarioNES was developed by Gary Boyes; LuigiNES is independent.\n"
            f"Single-file NES emulator, Python {sys.version.split()[0]}\n\n"
            f"Display locked at {TARGET_FPS:.0f} FPS NTSC.\n"
            "CPU: all 256 6502 opcodes, cycle-counted.\n"
            "PPU: 256x240, loopy scrolling, sprite-0 hit.\n"
            "APU: 2A03 synthesis, audio-locked pacing.\n"
            "Mappers: " + ", ".join(str(m) for m in sorted(_MAPPER_TABLE)) + ".")

    def _controls_help(self):
        messagebox.showinfo(
            "Controls",
            "D-Pad  : Arrow keys (or WASD)\n"
            "A      : X\n"
            "B      : Z\n"
            "Start  : Enter\n"
            "Select : Shift\n\n"
            "Pause  : P  (or Pause key)\n"
            "Reset  : Ctrl+R\n"
            "Open   : Ctrl+O")

    def _quit(self):
        self._closing = True
        self.running = False
        if self._after_job is not None:
            try:
                self.root.after_cancel(self._after_job)
            except tk.TclError:
                pass
            self._after_job = None
        for job in self._release_jobs.values():
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        self._release_jobs.clear()
        if self.audio:
            self.audio.stop()
        _win_timer_resolution(False)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # -- pacing ---------------------------------------------------------------
    def _frame_period(self):
        """60.00 Hz nudged by up to +-0.5% to hold the audio queue at target.
        That closes the loop between perf_counter and the sound card's own
        crystal, so the buffer can neither creep up (lag) nor drain (pops)."""
        a = self.audio
        if a is None or not a.active:
            return FRAME_SEC
        err = (a.buffered() - a.TARGET_FILL) / a.TARGET_FILL
        err = max(-1.0, min(1.0, err))
        return FRAME_SEC * (1.0 + 0.005 * err)

    def _schedule_tick(self, delay):
        if self._closing:
            return
        try:
            self._after_job = self.root.after(delay, self._frame_tick)
        except tk.TclError:
            self._after_job = None
            self.running = False

    def _frame_tick(self):
        self._after_job = None
        if self._closing:
            return
        now = time.perf_counter()
        if not (self.nes and self.running and not self.paused):
            self._frame_clock = now + FRAME_SEC
            self._schedule_tick(16)
            return
        remaining = self._frame_clock - now
        if remaining > _SPIN_SEC:
            self._schedule_tick(max(1, int((remaining - _SPIN_SEC) * 1000)))
            return
        while time.perf_counter() < self._frame_clock:
            pass                                # sub-2 ms spin for exact edge

        frames = 0
        try:
            while True:
                self.nes.run_frame()
                frames += 1
                self._frame_clock += self._frame_period()
                now = time.perf_counter()
                if now < self._frame_clock or frames >= _MAX_CATCHUP_FRAMES:
                    break
            self._blit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.running = False
            self.status_var.set(f"emulation error: {e}")
        if self.nes is not None and self.nes.cpu.jammed and self.running:
            self.status_var.set(f"CPU JAM at ${self.nes.cpu.pc:04X}")
            self.running = False
        now = time.perf_counter()
        if now - self._frame_clock > 0.1:      # long stall: resync, don't race
            self._frame_clock = now + FRAME_SEC
        self._update_fps(frames)
        delay = self._frame_clock - now - _SPIN_SEC
        self._schedule_tick(max(1, int(delay * 1000)) if delay > 0 else 1)

    def _update_fps(self, frames):
        self._fps_frames += frames
        self._fps_draws += 1
        dt = time.perf_counter() - self._fps_t0
        if dt >= 1.0:
            self.fps = self._fps_frames / dt
            draws = self._fps_draws / dt
            self._fps_frames = self._fps_draws = 0
            self._fps_t0 = time.perf_counter()
            self.status_fps_var.set(f"{self.fps:.1f} fps  (draw {draws:.0f})")

    # -- video ------------------------------------------------------------
    def _blit(self):
        nes = self.nes
        if nes is None:
            return
        fb = nes.ppu.framebuffer
        if self._blit_mode == "ppm":
            r, g, b = RGB_PLANES[nes.ppu.emphasis]
            rgb = self._rgb
            rgb[0::3] = fb.translate(r)
            rgb[1::3] = fb.translate(g)
            rgb[2::3] = fb.translate(b)
            try:
                self._src.configure(data=self._ppm_header + rgb, format="PPM")
                self.photo.tk.call(str(self.photo), "copy", str(self._src),
                                   "-zoom", SCALE, SCALE)
                return
            except tk.TclError:
                self._blit_mode = "hex"   # Tk without binary PPM -data support
        pal = NES_PALETTE_HEX
        rows = []
        for y in range(SCREEN_H):
            off = y * SCREEN_W
            rows.append("{" + " ".join(pal[fb[off + x] & 0x3F]
                                       for x in range(SCREEN_W)) + "}")
        # One row per Tcl command avoids oversized interpreter arguments.
        for y, row in enumerate(rows):
            self._src.put(row, to=(0, y))
        self.photo.tk.call(str(self.photo), "copy", str(self._src),
                           "-zoom", SCALE, SCALE)


def opcode_selftest():
    """Exercise every 2A03 opcode handler without opening Tk or loading a ROM.

    This checks decode coverage, execution, and basic cycle accounting.
    It is not a replacement for nestest or blargg accuracy ROMs.
    """
    import types
    errors = []
    entries = {}
    for line in _CPU_SPEC.strip().splitlines():
        mnemonic, *specs = line.split()
        for spec in specs:
            mode, opcode, cycles = spec.split(":")
            entries[int(opcode, 16)] = (mnemonic, mode, int(cycles))
    if set(entries) != set(range(256)):
        raise AssertionError("Incomplete 256-opcode decode table")
    for opcode in range(256):
        mnemonic, mode, expected = entries[opcode]
        memory = bytearray(65536)
        memory[0x8000] = opcode
        memory[0x8001] = 0x00
        memory[0x8002] = 0x40
        memory[0xFFFC] = 0x00
        memory[0xFFFD] = 0x80
        memory[0xFFFE] = 0x00
        memory[0xFFFF] = 0x90
        memory[0xFFFA] = 0x00
        memory[0xFFFB] = 0x90
        # Use a non-I/O absolute address for instruction tests.
        memory[0x8002] = 0x60
        ram = bytearray(2048)
        def read(addr):
            addr &= 0xFFFF
            return ram[addr & 0x7FF] if addr < 0x2000 else memory[addr]
        def write(addr, val):
            addr &= 0xFFFF
            if addr < 0x2000:
                ram[addr & 0x7FF] = val & 255
            else:
                memory[addr] = val & 255
        cpu = types.SimpleNamespace(a=0x12, x=0, y=0, sp=0xFD,
            p=0x24, pc=0x8001, cycles=7, ipoll_i=0, ipoll_cycle=-1,
            jammed=False)
        handlers = _CPU_FACTORY(cpu, read, write, ram, _ZN)
        try:
            elapsed = handlers[opcode]()
            if not isinstance(elapsed, int) or elapsed < expected:
                raise AssertionError(f"invalid cycles {elapsed}, expected >= {expected}")
            if not (0 <= cpu.pc <= 0xFFFF and 0 <= cpu.sp <= 0xFF
                    and 0 <= cpu.a <= 0xFF and 0 <= cpu.x <= 0xFF
                    and 0 <= cpu.y <= 0xFF):
                raise AssertionError("register out of range")
            if (mnemonic == "JAM") != bool(cpu.jammed):
                raise AssertionError("JAM state mismatch")
        except Exception as exc:
            errors.append(f"${opcode:02X} {mnemonic} {mode}: {exc}")
    if errors:
        raise AssertionError("Opcode test failed:\n" + "\n".join(errors))
    print("PASS: all 256 NES 2A03 opcodes decode and execute (smoke test)")
    return True


def main():
    if "--test-opcodes" in sys.argv:
        opcode_selftest()
        return 0
    if sys.version_info < (3, 14):
        print(f"{WINDOW_TITLE} requires Python 3.14+, got {sys.version}",
              file=sys.stderr)
        return 1
    rom = sys.argv[1] if len(sys.argv) > 1 else None
    root = tk.Tk()
    EmuApp(root, rom)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())