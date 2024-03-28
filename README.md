# Unity 디지털 트윈 환경에서의 자율주행 DQN
<ul>
  <li>
    Unity + ML-Agent Toolkit 을 활용한 실세계와 유사한 가상 도로 환경 구축
    <ul>
      <li>
        왕복 4차선 도로
      </li>
      <li>
        Box Collider를 이용한 도로 연석 구현
      </li>
      <li>
        Ray Perception Sensor, Camera Sensor를 부착한 차량 에이전트 활용
      </li>
      <li>
        Unity Asset :  Stylized Vehicles Pack - FREE (에이전트, 장애물), Cartoon Road Constructor (도로)
      </li>
    </ul>
  </li>
  <li>
    Python API를 활용한 DQN(Deep Q-Network) 구조 설계
    <ul>
      <li>
        Using: PyTorch, CuPy, Numpy, CUDA, Anaconda
      </li>
      <li>
        Input(State Space): Image queue, Ray distance list
      </li>
      <li>
        Action space: 15개의 discrete actions
      </li>
      <li>
        Unity Asset: Stylized Vehicles Pack - FREE (에이전트, 장애물), Cartoon Road Constructor (도로)
      </li>
    </ul>
  </li>
  <li>
    구현 목표: 구축 환경에서 심층 강화학습으로 차션 유지, 장애물 회피 주행하며 빠른 속도로 한 바퀴 주행
  </li>
</ul>


## 3000 episode 학습 결과 (4배속)   
![ezgif com-video-to-gif (1)](https://github.com/dd-jero/Autonomous-driving-DQN-Deep-Q-Network-in-Unity-digital-twin-environment/assets/107921434/81b610aa-012a-4ddc-8270-60d290a572ba)   
- 평균 주행 속도 10.3m/s (최대 가능 속도 12.5m/s)   
[종설 포스터_권혁규김형석이재영_최종.pptx](https://github.com/dd-jero/Comprehensive-Information-and-Communication-Design/files/11925421/_._.pptx)
