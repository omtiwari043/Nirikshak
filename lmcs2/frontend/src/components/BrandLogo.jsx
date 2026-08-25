export default function BrandLogo({ className = "", priority = false }) {
  return (
    <img
      src="/brand/nirikshak-logo.jpeg"
      alt="NIRIKSHAK"
      className={`object-contain ${className}`}
      loading={priority ? "eager" : "lazy"}
    />
  );
}
