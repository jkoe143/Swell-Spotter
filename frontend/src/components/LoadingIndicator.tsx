type Props = {
  isLoading: boolean;
};

export function LoadingIndicator({ isLoading }: Props) {
  if (!isLoading) return null;

  return (
    <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-blue-500 text-white px-4 py-2 rounded-full shadow-lg text-sm font-medium flex items-center gap-2">
      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
      Loading tiles...
    </div>
  );
}
