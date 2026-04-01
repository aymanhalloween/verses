import { motion } from "framer-motion";

export function Background() {
  return (
    <div className="fixed inset-0 w-full h-full z-0 overflow-hidden pointer-events-none bg-[#0D0D0F]">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 2.5, ease: "easeOut" }}
        className="absolute inset-0"
      >
        <img
          src="/images/mosque-bg.webp"
          alt=""
          className="w-full h-full object-cover object-top"
          style={{ filter: "brightness(0.45) saturate(0.8)" }}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-black/10 via-[#0D0D0F]/60 to-[#0D0D0F]" />
        <div className="absolute inset-0 bg-gradient-to-t from-[#0D0D0F] via-transparent to-transparent" style={{ height: "50%", top: "50%" }} />
      </motion.div>
    </div>
  );
}
