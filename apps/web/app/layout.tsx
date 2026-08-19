export default function RootLayout({children}: {children: React.ReactNode}) {
  return (
    <html lang="bn">
      <body>{children}</body>
    </html>
  );
}
