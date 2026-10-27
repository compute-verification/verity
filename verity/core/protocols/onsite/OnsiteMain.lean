import CoreProtocol.Cli
import CoreOnsite

def main : List String → IO UInt32
  | "bound" :: args => Core.Onsite.Bound.main Core.Onsite.protocol.id args
  | args => Core.Onsite.protocol.main args
