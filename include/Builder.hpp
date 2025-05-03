// #include "Gbuilder.cuh"
// #include <cstdint>
// #include <utility>
// #include <vector>

// namespace Gbuilder {
// template <typename T>
// class Builder {
//  public:
//   using data_type = T;
//   using id_type = uint32_t;

//   Builder() = delete;

//   explicit Builder(std::vector<data_type> const& base,
//                    std::vector<id_type> const& gt_ids)
//       : base_(base), gt_ids_(gt_ids) {}

//   Builder(Builder&& other) noexcept
//       : base_(std::move(other.base_)),
//         query_(std::move(other.query_)),
//         gt_ids_(std::move(other.gt_ids_)) {}

//   Builder(Builder& other) = delete;
//   Builder& operator=(Builder const&) = delete;
//   Builder& operator=(Builder&&) = delete;
//   ~Builder() = default;

//   template <uint32_t grid_size, uint32_t block_size, uint32_t gt_num,
//             uint32_t dim>
//   auto LaunchBuild() -> void;

//   auto SetGt(std::vector<id_type>&& gt_ids) -> void {
//     gt_ids_ = std::move(gt_ids);
//   }

//  private:
//   std::vector<data_type> base_;
//   std::vector<data_type> query_;
//   std::vector<id_type> gt_ids_;
// };
// }  // namespace Gbuilder
