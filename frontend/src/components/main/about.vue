<template>
    <el-card class="box-card" style="width:1000px">
    <template #header>
      <div class="card-header">
        <span style="font-size: 24px; font-weight: bold;">乐改-航空航天lacorrect-hkht：面向航空航天领域的文本纠错平台</span>
      </div>
    </template>
    <div class="text item">
      <div>
        &nbsp&nbsp&nbsp&nbsp&nbsp乐改-航空航天（lacorrect-hkht）是“语析LAGroup”（可关注我们的同名公众号）开发的文本纠错平台。本平台聚焦航空航天文档，提供纠错功能。我们将持续积累最新技术，提高系统性能。
      </div>
      <div>
        &nbsp
      </div>
      <div>
        &nbsp&nbsp&nbsp&nbsp&nbsp从2020年起，“语析LAGroup”开始研究文本纠错（文本校对）任务。我们针对二语学习者文本和母语使用者文本，分别标注了一定规模的高质量文本纠错数据集。目前发表两篇顶会论文（NAACL-2022和EMNLP-2022）。我们连续四次在国内文本纠错大赛中取得第一名（CTC-2021、CGED-2021、WAIC-2022、CAIL-2022）。
      </div>
    </div>
    <div class="text item">
      系统开发人员：蒋浩辰，乔子恒
      <div>
        &nbsp
      </div>
      <div>
        核心技术支持：章岳、周厚全
      </div>
      <div>
        &nbsp
      </div>
      <div>
        指导老师：李正华
      </div>
      <div>
        &nbsp
      </div>
      <div>
        若您发现问题、或者有建议，请联系蒋浩辰（jhcnlp@qq.com）进行反馈，我们会真诚表达谢意！
      </div>
      <div>
        &nbsp
      </div>
      <div>
        &nbsp
      </div>
      <div>
        &nbsp
      </div>
      <div>
        &nbsp
      </div>
      <div>
        &nbsp
      </div>
      <div>
        附获奖证书：
      </div>
      <div>
        &nbsp
      </div>
      <div>
        <el-image
        style="width: 600px; height: 400px;"
        :src="url3"
        />
        &nbsp&nbsp
        <el-image
          style="width: 300px; height: 400px;"
          :src="url4"
        />
      </div>
      <div>
        &nbsp
      </div>
      <div>
        <el-image
          style="width: 600px; height: 400px;"
          :src="url1"
        />
        &nbsp&nbsp
        <el-image
        style="width: 300px; height: 400px;"
        :src="url2"
        />
      </div>
    </div>
  </el-card>
</template>
  
<script setup>
  const url1 = require('../../../src/assets/证书/蜜度文本纠错获奖证书.jpg')
  const url2 = require('../../../src/assets/证书/法研杯2022获奖证书.jpg')
  const url3 = require('../../../src/assets/证书/第三届中国AI+文本纠错获奖.jpg')
  const url4 = require('../../../src/assets/证书/CGED2021获奖证书.jpg')
  
</script>

<style lang="less" scoped>
</style>